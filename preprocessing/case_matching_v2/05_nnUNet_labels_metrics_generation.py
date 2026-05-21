"""
05_nnUNet_labels_metrics_generation.py — nnUNet inference + AAA metrics (v2, CT only)
=======================================================================================

Processes CT NIfTIs in training_subset_v2/imagesTr/.
All other modalities (MR, US, XR, PET, XA, …) are skipped — the nnUNet model was
trained on CTA and segmentation on other modalities is not supported.

Produces per CT scan:
  labelsTr/{stem}.nii.gz    — nnUNet segmentation (0=bg, 1=lumen, 2=ILT)
  distMapsTr/{stem}.nii.gz  — lumen-to-wall distance map (mm)
  metricsTr/{stem}.json     — volumetric + ILT thickness metrics

All outputs use the same file stem as the input NIfTI, e.g.:
  2637442_AUMC_s001_CT.nii.gz → 2637442_AUMC_s001_CT.json

Usage
-----
python 05_nnUNet_labels_metrics_generation.py \
    --input    /deepstore/.../training_subset_v2/imagesTr \
    --output-dir /deepstore/.../training_subset_v2 \
    --checkpoint-dir /path/to/nnunet_model \
    [--device cpu|cuda]
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# =========================================================
# METRICS + DISTANCE MAP
# =========================================================

def compute_metrics_and_distance(mask_path: Path, stem: str, output_base: Path) -> dict:
    """
    Derive aneurysm morphology metrics and voxel-wise distance maps from
    an nnUNet segmentation mask.

    Labels assumed: 0=background, 1=lumen, 2=ILT
    """
    img    = nib.load(str(mask_path))
    data   = img.get_fdata().astype(np.uint8)
    affine = img.affine
    header = img.header
    spacing = header.get_zooms()
    voxvol  = float(np.prod(spacing))

    lumen_mask    = data == 1
    ilt_mask      = data == 2
    total_aaa_mask = lumen_mask | ilt_mask

    # Distance transform: distance from each voxel to the lumen surface
    dist_map        = distance_transform_edt(~lumen_mask, sampling=spacing)
    dist_map_masked = dist_map * total_aaa_mask

    max_ilt = float(np.max(dist_map[ilt_mask])) if np.any(ilt_mask) else 0.0
    avg_ilt = float(np.mean(dist_map[ilt_mask])) if np.any(ilt_mask) else 0.0

    metrics = {
        "volume_lumen_ml":           float(np.sum(lumen_mask)     * voxvol / 1000.0),
        "volume_ilt_ml":             float(np.sum(ilt_mask)        * voxvol / 1000.0),
        "volume_total_ml":           float(np.sum(total_aaa_mask)  * voxvol / 1000.0),
        "max_lumen_to_wall_dist_mm": max_ilt,
        "avg_lumen_to_wall_dist_mm": avg_ilt,
    }

    dist_dir = output_base / "distMapsTr"
    dist_dir.mkdir(exist_ok=True)
    nib.save(
        nib.Nifti1Image(dist_map_masked.astype(np.float32), affine, header),
        str(dist_dir / f"{stem}.nii.gz"),
    )

    met_dir = output_base / "metricsTr"
    met_dir.mkdir(exist_ok=True)
    with open(met_dir / f"{stem}.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# =========================================================
# nnUNet INFERENCE
# =========================================================

def run_prediction(
    predictor: "nnUNetPredictor",
    scan_path: Path,
    output_base: Path,
) -> None:
    """Run nnUNet on one CT NIfTI and compute downstream metrics."""
    stem = scan_path.name.replace(".nii.gz", "")

    labels_dir = output_base / "labelsTr"
    labels_dir.mkdir(exist_ok=True)
    final_mask_path = labels_dir / f"{stem}.nii.gz"

    tmp_dir = output_base / f"_tmp_{stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    predictor.predict_from_files(
        [[str(scan_path)]],
        str(tmp_dir),
        save_probabilities=False,
        overwrite=True,
    )

    pred_files = list(tmp_dir.glob("*.nii.gz"))
    if pred_files:
        shutil.move(str(pred_files[0]), str(final_mask_path))
        compute_metrics_and_distance(final_mask_path, stem, output_base)
    else:
        print(f"[!] nnUNet produced no output for {stem}")

    shutil.rmtree(tmp_dir)
    print(f"[nnUNet] Done: {stem}")


# =========================================================
# ENTRY POINT
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="nnUNet v2 inference + AAA metrics for all CT scans")
    ap.add_argument("--input",          required=True,  help="imagesTr directory")
    ap.add_argument("--output-dir",     required=True,  help="training_subset_v2 root")
    ap.add_argument("--checkpoint-dir", required=True,  help="nnUNet trained model folder")
    ap.add_argument("--device",         default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)
    ckpt_dir    = Path(args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_scans = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.name.endswith("_CT.nii.gz")
    )
    print(f"Found {len(all_scans)} CT scan(s) to process")

    trainer_dirs = [p for p in ckpt_dir.rglob("*") if p.is_dir() and "nnUNetTrainer" in p.name]
    if not trainer_dirs:
        raise RuntimeError(f"No nnUNetTrainer folder found under {ckpt_dir}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device(args.device),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=trainer_dirs[0],
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    labels_dir = output_dir / "labelsTr"
    for scan_path in all_scans:
        stem = scan_path.name.replace(".nii.gz", "")
        if (labels_dir / f"{stem}.nii.gz").exists():
            print(f"[SKIP] {stem} already segmented")
            continue
        run_prediction(predictor, scan_path, output_dir)


if __name__ == "__main__":
    main()
