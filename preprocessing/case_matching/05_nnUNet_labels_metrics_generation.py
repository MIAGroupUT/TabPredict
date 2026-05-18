"""
===========================================================
nnUNetv2 Inference + Distance Map + Metrics Pipeline
===========================================================

This script performs:

1. nnUNet inference on input CT scans
2. Post-processing of segmentation masks
3. Computation of aneurysm-derived metrics
4. Generation of voxel-wise distance maps
5. Saving results in a structured dataset format

-----------------------------------------------------------
Expected input naming convention:
-----------------------------------------------------------
{p_id}_{site_abbr}.nii.gz

-----------------------------------------------------------
Outputs:
-----------------------------------------------------------
output_dir/
    labelsTr/        -> predicted segmentation masks
    distMapsTr/      -> distance maps (ILT / lumen thickness)
    metricsTr/       -> quantitative metrics JSON per patient

-----------------------------------------------------------
Execution example:
-----------------------------------------------------------
python3 03_nnUNet_labels_metrics_generation.py \
    --input /path/to/scans \
    --output-dir /path/to/output \
    --checkpoint-dir /path/to/nnunet_model \
    --device cuda
"""

import argparse
import shutil
from pathlib import Path
import torch
import nibabel as nib
import numpy as np
import json
from scipy.ndimage import distance_transform_edt
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def compute_metrics_and_distance(mask_path: Path, case_id: str, output_base: Path):
    """
    Compute aneurysm morphology metrics and voxel-wise distance maps.

    This function derives:
    - volumetric measurements (lumen, ILT, total AAA)
    - lumen-to-wall distance maps
    - ILT thickness statistics

    Parameters
    ----------
    mask_path : Path
        Path to predicted segmentation mask (nnUNet output).

    case_id: str
        Patient identifier extracted from filename.

    output_base : Path
        Base output directory where results are stored.

    Outputs
    -------
    distMapsTr/
        {case_id}.nii.gz → masked distance map (mm)

    metricsTr/
        {case_id}.json → scalar anatomical metrics

    Notes
    -----
    Segmentation labels assumed:
        0 = background
        1 = lumen
        2 = intraluminal thrombus (ILT)
    """

    img = nib.load(str(mask_path))
    data = img.get_fdata().astype(np.uint8)
    affine = img.affine
    header = img.header
    spacing = header.get_zooms()

    # ----------------------------
    # Volume calculations
    # ----------------------------
    voxvol = np.prod(spacing)

    lumen_mask = (data == 1)
    ilt_mask = (data == 2)
    total_aaa_mask = (data == 1) | (data == 2)

    # ----------------------------
    # Distance transform
    # ----------------------------
    dist_map = distance_transform_edt(~lumen_mask, sampling=spacing)
    dist_map_masked = dist_map * total_aaa_mask

    # ----------------------------
    # ILT thickness metrics
    # ----------------------------
    max_ilt_thickness = 0.0
    avg_ilt_thickness = 0.0

    if np.any(ilt_mask):
        max_ilt_thickness = float(np.max(dist_map[ilt_mask]))
        avg_ilt_thickness = float(np.mean(dist_map[ilt_mask]))

    metrics = {
        "volume_lumen_ml": float(np.sum(lumen_mask) * voxvol / 1000.0),
        "volume_ilt_ml": float(np.sum(ilt_mask) * voxvol / 1000.0),
        "volume_total_ml": float(np.sum(total_aaa_mask) * voxvol / 1000.0),
        "max_lumen_to_wall_dist_mm": max_ilt_thickness,
        "avg_lumen_to_wall_dist_mm": avg_ilt_thickness
    }

    # ----------------------------
    # Save distance map
    # ----------------------------
    dist_dir = output_base / "distMapsTr"
    dist_dir.mkdir(exist_ok=True)

    dist_nifti = nib.Nifti1Image(dist_map_masked.astype(np.float32), affine, header)
    nib.save(dist_nifti, str(dist_dir / f"{case_id}.nii.gz"))

    # ----------------------------
    # Save metrics
    # ----------------------------
    met_dir = output_base / "metricsTr"
    met_dir.mkdir(exist_ok=True)

    with open(met_dir / f"{case_id}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_prediction(predictor, scan_path: Path, output_base: Path, calculate_metrics: bool = True):
    """
    Run nnUNet inference for a single scan and optionally compute metrics.

    Pipeline steps:
    ---------------
    1. Extract patient ID from filename
    2. Run nnUNet prediction
    3. Move predicted mask to labelsTr/
    4. Compute distance maps + metrics (optional)
    5. Cleanup temporary files

    Parameters
    ----------
    predictor : nnUNetPredictor
        Initialized nnUNet inference model.

    scan_path : Path
        Input CT scan file.

    output_base : Path
        Output dataset directory.

    calculate_metrics : bool
        If True, compute distance maps and anatomical metrics.
    """

    p_id, site_abbr = scan_path.name.replace(".nii.gz", "").split("_")

    labels_dir = output_base / "labelsTr"
    labels_dir.mkdir(exist_ok=True)

    final_mask_path = labels_dir / f"{p_id}_{site_abbr}.nii.gz"

    tmp_out_dir = output_base / f"tmp_{p_id}"
    tmp_out_dir.mkdir(exist_ok=True, parents=True)

    predictor.predict_from_files(
        [[str(scan_path)]],
        str(tmp_out_dir),
        save_probabilities=False,
        overwrite=True
    )

    pred_files = list(tmp_out_dir.glob("*.nii.gz"))

    if pred_files:
        shutil.move(str(pred_files[0]), str(final_mask_path))

        if calculate_metrics:
            case_id = f"{p_id}_{site_abbr}"
            compute_metrics_and_distance(final_mask_path, case_id, output_base)

    shutil.rmtree(tmp_out_dir)

    print(f"[nnUNet] Processed {p_id}_{site_abbr}")


def main():
    """
    Entry point for nnUNet inference pipeline.

    Responsibilities:
    -----------------
    - Parse CLI arguments
    - Load nnUNet model checkpoint
    - Collect input scans
    - Run inference sequentially
    - Skip already processed cases
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Resolve input scans
    # ----------------------------
    if input_path.is_file():
        scans = [input_path]
    elif input_path.is_dir():
        scans = [
            p for p in input_path.iterdir()
            if p.is_file() and any(str(p).endswith(ext) for ext in (".nii", ".nii.gz", ".mhd"))
        ]
    else:
        raise RuntimeError(f"Input path not found: {input_path}")

    print(f"[nnUNet] Found {len(scans)} scan(s)")

    # ----------------------------
    # Locate trained model folder
    # ----------------------------
    trainer_dirs = [
        p for p in ckpt_dir.rglob("*")
        if p.is_dir() and "nnUNetTrainer" in p.name
    ]

    if not trainer_dirs:
        raise RuntimeError(f"No nnUNetTrainer folder found under {ckpt_dir}")

    trainer_path = trainer_dirs[0]

    # ----------------------------
    # Initialize predictor
    # ----------------------------
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device(args.device),
        verbose=False,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=trainer_path,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    # ----------------------------
    # Run inference
    # ----------------------------
    for scan_path in scans:

        p_id, site_abbr = scan_path.name.replace(".nii.gz", "").split("_")

        if (output_dir / "labelsTr" / f"{p_id}_{site_abbr}.nii.gz").exists():
            print(f"Skipping {p_id}_{site_abbr}, already segmented.")
            continue

        run_prediction(predictor, scan_path, output_dir)


if __name__ == "__main__":
    main()