"""
06_crop_full_body_scans.py — Crop CT volumes to the AAA region using nnUNet masks
==================================================================================

For each CT in training_subset_v1/imagesTr/ that has a corresponding nnUNet
segmentation mask in labelsTr/, determines the axial (superior-inferior) extent
of the segmented aortic region and crops the volume to that extent plus a
configurable padding fraction.

Also crops the matching labelsTr/ and distMapsTr/ files in-place so that all
three paired files remain spatially consistent after cropping.

Motivation
----------
Full-body CTAs (head-to-toe) contain large empty axial regions above the
renal arteries and below the iliac bifurcation.  The MONAI pipeline resamples
to (1.5, 1.5, 3) mm and then pads/crops to (224, 224, 160) voxels.  The
target axial physical extent is 160 × 3 mm = 480 mm.  Without cropping, the
aortic sac occupies only a small fraction of the 160-slice target.  After
cropping this script guarantees the axial physical extent is ≥ 480 mm, so the
aorta fills the full target volume and zero-padding is minimal.

Only the axial (S-I) axis is cropped.  X/Y dimensions are left unchanged.

Pipeline position
-----------------
    04_data_organizing              → imagesTr/
    05_nnUNet_labels_metrics_gen... → labelsTr/, distMapsTr/, metricsTr/
    06_crop_full_body_scans (this)  → crops imagesTr/, labelsTr/, distMapsTr/
    07_generate_data_split          → dataset_splits.json

Usage
-----
    # Preview — no files written
    python 06_crop_full_body_scans.py \\
        --input-dir /deepstore/.../training_subset_v1 --dry-run

    # Apply
    python 06_crop_full_body_scans.py \\
        --input-dir /deepstore/.../training_subset_v1

    # Custom padding / skip threshold
    python 06_crop_full_body_scans.py \\
        --input-dir /deepstore/.../training_subset_v1 \\
        --pad-frac 0.10 --skip-threshold 0.80
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation


# =========================================================
# ORIENTATION HELPERS
# =========================================================

def find_axial_axis(affine: np.ndarray) -> int:
    """
    Return the array dimension index that corresponds to the axial (S-I) direction.

    Uses nibabel's io_orientation to map each array axis to a world (R/A/S) axis.
    World axis 2 is the S-I axis (superior-inferior = axial in standard CT).
    Falls back to axis 2 (last dimension, the conventional NIfTI slice axis) if
    the affine does not assign S-I unambiguously.
    """
    orient = io_orientation(affine)
    for img_dim, (world_dim, _) in enumerate(orient):
        if int(world_dim) == 2:
            return int(img_dim)
    return 2


# =========================================================
# CROP HELPER
# =========================================================

def crop_volume(
    img_path: Path,
    axial_ax: int,
    crop_start: int,
    crop_end: int,
    out_path: Path | None = None,
) -> None:
    """
    Load a NIfTI, crop along axial_ax from crop_start to crop_end.

    Writes to out_path if given, otherwise overwrites img_path in-place.
    Updates the affine origin so world coordinates of surviving voxels are
    unchanged.  Preserves the original data type (int16 for CT, uint8 for labels,
    float32 for distance maps).
    """
    img  = nib.load(str(img_path))
    data = img.get_fdata(dtype=np.float32)

    slices = [slice(None)] * 3
    slices[axial_ax] = slice(crop_start, crop_end)
    cropped = data[tuple(slices)]

    # Shift origin: voxel crop_start along axial_ax maps to the new voxel 0
    new_affine = img.affine.copy()
    new_affine[:3, 3] = img.affine[:3, 3] + img.affine[:3, axial_ax] * crop_start

    new_header = img.header.copy()
    new_header.set_data_shape(cropped.shape)

    dest = out_path if out_path is not None else img_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            cropped.astype(img.get_data_dtype()),
            new_affine,
            new_header,
        ),
        str(dest),
    )


# =========================================================
# PER-SCAN LOGIC
# =========================================================

def process_scan(
    ct_path: Path,
    output_base: Path,
    pad_frac: float,
    skip_threshold: float,
    dry_run: bool,
    suffix: str = "",
) -> str:
    """
    Crop one CT scan and its paired label + distance-map files.

    Skips if:
      - no nnUNet label mask exists (step 05 not yet run for this scan)
      - the mask is empty (segmentation failure)
      - the mask already covers >= skip_threshold of the axial extent
        (the scan is already aorta-focused, no crop needed)

    Output subdirectory names are imagesTr{suffix}/, labelsTr{suffix}/,
    distMapsTr{suffix}/ — all under output_base.  Empty suffix = in-place overwrite.

    Returns a status string (may be multi-line) for console output.
    """
    stem       = ct_path.name.replace(".nii.gz", "")
    label_path = output_base / "labelsTr"   / f"{stem}.nii.gz"
    dist_path  = output_base / "distMapsTr" / f"{stem}.nii.gz"

    if not label_path.exists():
        return f"[SKIP] {stem}\n        no label mask — run step 05 first"

    # Load label as uint8 to find axial extent of the aorta
    lbl_img  = nib.load(str(label_path))
    lbl_data = np.asarray(lbl_img.dataobj, dtype=np.uint8)
    affine   = lbl_img.affine

    axial_ax = find_axial_axis(affine)
    total    = lbl_data.shape[axial_ax]
    shape_in = lbl_data.shape

    # Collapse X and Y so we get a 1-D mask along the axial axis
    other_axes   = tuple(i for i in range(3) if i != axial_ax)
    aorta_slices = np.any(lbl_data > 0, axis=other_axes)
    nonzero      = np.where(aorta_slices)[0]

    if len(nonzero) == 0:
        return f"[SKIP] {stem}\n        empty segmentation mask"

    z_min  = int(nonzero[0])
    z_max  = int(nonzero[-1])
    extent = z_max - z_min + 1

    axial_voxel_mm = float(np.linalg.norm(affine[:3, axial_ax]))

    # Skip scans that are already tightly cropped around the aorta
    coverage = extent / total
    if coverage >= skip_threshold:
        return (
            f"[SKIP] {stem}\n"
            f"        aorta {z_min}–{z_max} ({extent} sl) covers "
            f"{coverage:.1%} of {total} sl ({total * axial_voxel_mm:.0f} mm) "
            f"— already focused (threshold {skip_threshold:.0%})"
        )

    pad        = max(1, round(pad_frac * extent))
    crop_start = max(0, z_min - pad)
    crop_end   = min(total, z_max + pad + 1)

    # Ensure the cropped axial extent covers at least 160 × 3 mm = 480 mm in
    # physical space so that after resampling to 3 mm z-spacing the volume has
    # ≥ 160 slices.  Without this, SpatialPadd fills most of the target volume
    # with zeros and Merlin sees very little anatomy.
    min_voxels     = int(np.ceil(480.0 / axial_voxel_mm))
    current_extent = crop_end - crop_start
    min_enforced   = current_extent < min_voxels
    if min_enforced:
        deficit    = min_voxels - current_extent
        expand_lo  = deficit // 2
        expand_hi  = deficit - expand_lo
        crop_start = max(0, crop_start - expand_lo)
        crop_end   = min(total, crop_end + expand_hi)
        # If one boundary was hit, compensate on the other side
        if crop_start == 0:
            crop_end = min(total, min_voxels)
        if crop_end == total:
            crop_start = max(0, total - min_voxels)

    new_size = crop_end - crop_start
    phys_in  = total   * axial_voxel_mm
    phys_out = new_size * axial_voxel_mm

    new_shape = list(shape_in)
    new_shape[axial_ax] = new_size
    new_shape = tuple(new_shape)

    def fmt_shape(s):
        return f"{s[0]}×{s[1]}×{s[2]}"

    out_images = output_base / f"imagesTr{suffix}"
    out_labels = output_base / f"labelsTr{suffix}"
    out_dists  = output_base / f"distMapsTr{suffix}"
    out_ct     = out_images / ct_path.name
    out_label  = out_labels / label_path.name
    out_dist   = out_dists  / dist_path.name

    has_dist   = dist_path.exists()
    files_line = f"imagesTr{suffix}/ + labelsTr{suffix}/"
    if has_dist:
        files_line += f" + distMapsTr{suffix}/"

    min_note = f" [min-extent enforced: needed {min_voxels} vox]" if min_enforced else ""
    tag      = "DRY " if dry_run else "CROP"
    status = (
        f"[{tag}] {stem}\n"
        f"        in : {fmt_shape(shape_in)}  axial {total} sl  {phys_in:.0f} mm\n"
        f"        out: {fmt_shape(new_shape)}  axial {new_size} sl  {phys_out:.0f} mm{min_note}\n"
        f"        aorta z={z_min}–{z_max} ({extent} sl), pad ±{pad}, crop [{crop_start}:{crop_end}]\n"
        f"        {'would write' if dry_run else 'wrote'}: {files_line}"
    )

    if not dry_run:
        crop_volume(ct_path,    axial_ax, crop_start, crop_end, out_path=out_ct)
        crop_volume(label_path, axial_ax, crop_start, crop_end, out_path=out_label)
        if has_dist:
            crop_volume(dist_path, axial_ax, crop_start, crop_end, out_path=out_dist)

    return status


# =========================================================
# ENTRY POINT
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crop CTs to the nnUNet-detected aortic region (axial axis only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input-dir", required=True,
        help="training_subset root to read from (must contain imagesTr/ and labelsTr/)",
    )
    ap.add_argument(
        "--suffix", default="",
        help=(
            "Suffix appended to output subdirectory names, e.g. '_cropped' produces "
            "imagesTr_cropped/, labelsTr_cropped/, distMapsTr_cropped/ inside --input-dir. "
            "Default is empty, which overwrites the input subdirectories in-place."
        ),
    )
    ap.add_argument(
        "--pad-frac", type=float, default=0.125,
        help=(
            "Padding as a fraction of the mask axial extent, applied on each side. "
            "0.125 adds ~12.5%% of the aorta height above and below the segmentation."
        ),
    )
    ap.add_argument(
        "--skip-threshold", type=float, default=0.85,
        help=(
            "Skip scans where the aortic mask already spans this fraction of the total "
            "axial range — these are already focused and do not need cropping."
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print planned crops without modifying any files.",
    )
    args = ap.parse_args()

    base       = Path(args.input_dir)
    suffix     = args.suffix
    images_dir = base / "imagesTr"
    labels_dir = base / "labelsTr"

    if not images_dir.exists():
        raise SystemExit(f"[!] imagesTr/ not found under {base}")
    if not labels_dir.exists():
        raise SystemExit(
            f"[!] labelsTr/ not found under {base} — run step 05 first"
        )

    ct_scans = sorted(p for p in images_dir.iterdir() if p.name.endswith("_CT.nii.gz"))
    print(f"Found {len(ct_scans)} CT scan(s) in {images_dir}")
    if suffix:
        print(f"Output suffix: '{suffix}'  →  imagesTr{suffix}/, labelsTr{suffix}/, distMapsTr{suffix}/")
    else:
        print("No suffix — files will be overwritten in-place.")
    if args.dry_run:
        print("DRY RUN — no files will be modified.")
    print()

    n_cropped = n_skipped = 0
    for ct_path in ct_scans:
        status = process_scan(
            ct_path,
            output_base=base,
            pad_frac=args.pad_frac,
            skip_threshold=args.skip_threshold,
            dry_run=args.dry_run,
            suffix=suffix,
        )
        print(status)
        if "[CROP]" in status:
            n_cropped += 1
        else:
            n_skipped += 1

    print(f"\n{'=' * 62}")
    if args.dry_run:
        print(f"DRY RUN complete — would crop {n_cropped}, skip {n_skipped}")
    else:
        print(f"DONE — cropped {n_cropped}, skipped {n_skipped}")
    print(f"{'=' * 62}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import sys

    INPUT_DIR = "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2"

    # Inject --input-dir when not supplied (e.g. python 06_crop_full_body_scans.py --suffix _cropped)
    if "--input-dir" not in sys.argv:
        sys.argv += ["--input-dir", INPUT_DIR]

    main()
