import os
import SimpleITK as sitk
from pathlib import Path
import re
import pydicom
import numpy as np


def is_axial_orientation(ds):
    """
    Check if a DICOM slice is axial using ImageOrientationPatient.
    """
    try:
        iop = ds.ImageOrientationPatient
        row = np.array(iop[:3])
        col = np.array(iop[3:])
        normal = np.cross(row, col)

        # axial if normal ~ z-axis
        return abs(normal[2]) > 0.9
    except:
        return False

def is_axial_orientation(ds, tol=0.1):
    """
    Determine whether a DICOM series is in axial orientation.

    Parameters
    ----------
    ds : pydicom.Dataset
        DICOM dataset containing ImageOrientationPatient.

    tol : float
        Tolerance for floating point comparison.

    Returns
    -------
    bool
        True if orientation is approximately axial.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return False

    row = iop[:3]
    col = iop[3:]

    # Expected axial:
    # row ≈ [1, 0, 0]
    # col ≈ [0, 1, 0]
    return (
        abs(row[0] - 1) < tol and abs(row[1]) < tol and abs(row[2]) < tol and
        abs(col[1] - 1) < tol and abs(col[0]) < tol and abs(col[2]) < tol
    )


def score_dicom_series(dicom_dir, verbose=False):
    """
    Score a DICOM series to determine if it's the desired axial CTA scan.

    This function evaluates a series based on:
    - Orientation (axial preferred)
    - Series description keywords (arterial / CTA / contrast)
    - Slice thickness (thin slices preferred)
    - Number of slices (to avoid scouts)

    Parameters
    ----------
    dicom_dir : str or Path
        Path to a folder containing DICOM files of a single series.

    verbose : bool
        If True, prints debug information for inspection.

    Returns
    -------
    float
        Series score. Higher = more likely to be correct CTA scan.
        Returns -1 if invalid or unreadable.
    """

    dicom_dir = Path(dicom_dir)
    files = [f for f in dicom_dir.glob("*") if f.is_file()]

    if not files:
        return -1

    try:
        # Read first valid DICOM file
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    except Exception:
        return -1

    score = 0

    # ----------------------------
    # 1. Orientation (MOST IMPORTANT)
    # ----------------------------
    if is_axial_orientation(ds):
        score += 5
    else:
        score -= 3  # strong penalty if not axial

    # ----------------------------
    # 2. Series description keywords
    # ----------------------------
    desc = str(getattr(ds, "SeriesDescription", "")).lower()

    # Positive signals
    if "arter" in desc: score += 4        # arterial / arterieel
    if "cta" in desc: score += 4
    if "ce" in desc or "contrast" in desc: score += 2
    if "weke delen" in desc: score += 1   # MST specific

    # Weak axial hints
    if "ax" in desc: score += 2

    # Negative signals
    if any(x in desc for x in ["cor", "sag"]):
        score -= 5
    if any(x in desc for x in ["scout", "localizer", "topogram"]):
        score -= 10

    # ----------------------------
    # 3. Slice thickness
    # ----------------------------
    thickness = getattr(ds, "SliceThickness", None)
    if thickness is not None:
        try:
            thickness = float(thickness)
            if thickness <= 2.0:
                score += 2
            elif thickness > 5.0:
                score -= 2
        except:
            pass

    # ----------------------------
    # 4. Number of slices (VERY IMPORTANT)
    # ----------------------------
    n_slices = len(files)

    if n_slices > 100:
        score += 3
    elif n_slices < 20:
        score -= 5  # likely scout

    # ----------------------------
    # Debug output
    # ----------------------------
    if verbose:
        print(f"""
                Series: {dicom_dir}
                Description: {desc}
                Thickness:   {thickness}
                Slices:      {n_slices}
                Axial:       {is_axial_orientation(ds)}
                Score:       {score}
            """)

    return score

def find_best_dicom_folder(patient_folder: Path):
    """
    Select best DICOM series based on metadata scoring.
    """
    candidates = []

    for p in patient_folder.rglob("DICOM"):
        if p.is_dir():
            score = score_dicom_series(p, True)
            candidates.append((score, p))

    if not candidates:
        return None

    # sort by best score
    candidates.sort(key=lambda x: x[0], reverse=True)

    best_score, best_path = candidates[0]

    print(f"Selected {best_path} (score={best_score})")

    return best_path

def convert_dicom_folder_to_nifti(dicom_dir):
    """
    Convert a DICOM series folder into a SimpleITK image.

    Parameters
    ----------
    dicom_dir : str or Path
        Path to a folder containing a DICOM series (must include at least one valid series).

    Returns
    -------
    sitk.Image or None
        3D medical image if a valid DICOM series is found, otherwise None.

    Notes
    -----
    - Assumes each DICOM folder contains exactly one dominant series.
    - If multiple series exist, the first detected one is used.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        return None

    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)

    return reader.Execute()


def extract_patient_id(patient_folder_name: str):
    """
    Extract the patient ID from a folder name.

    This handles both clean IDs and CT-suffixed IDs.

    Parameters
    ----------
    patient_folder_name : str
        Name of the patient folder (e.g. "84", "0411_CT_1", "74826012_CT_1").

    Returns
    -------
    str
        Clean patient ID without CT suffix.

    Examples
    --------
    >>> extract_patient_id("0411_CT_1")
    "0411"

    >>> extract_patient_id("84")
    "84"
    """
    return patient_folder_name.split("_CT")[0]


def find_dicom_folder(root: Path):
    """
    Recursively locate the DICOM folder inside a patient directory.

    Expected structure (variable depth):
        {ID} / {any subfolder(s)} / DICOM

    Parameters
    ----------
    root : Path
        Root folder of a patient case.

    Returns
    -------
    Path or None
        Path to the DICOM folder if found, otherwise None.

    Notes
    -----
    - Searches recursively at any depth.
    - Matches folders named exactly 'DICOM' (case-insensitive).
    """
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "dicom":
            return p
    return None


def convert_dataset(parent_dir, output_dir):
    """
    Convert a multi-site nested DICOM dataset into NIfTI files.

    This function processes all sites and patients and produces a flat
    NIfTI dataset suitable for downstream ML pipelines.

    Expected input structure
    ------------------------
    parent_dir/
        RNST/
            0411_CT_1/4/DICOM
        AUMC/
            74826012_CT_1/9/DICOM
        MST/
            84/6/DICOM

    Output format
    -------------
    {patient_id}_{site}.nii.gz

    Examples
    --------
    0411_RNST.nii.gz
    74826012_AUMC.nii.gz
    84_MST.nii.gz

    Parameters
    ----------
    parent_dir : str or Path
        Root directory containing all site folders.

    output_dir : str or Path
        Destination directory for converted NIfTI files.

    Workflow
    ---------
    1. Iterate over all sites
    2. Iterate over patient folders
    3. Extract patient ID (remove optional _CT_x suffix)
    4. Locate DICOM folder recursively
    5. Convert DICOM → NIfTI
    6. Save as flattened dataset

    Assumptions
    -----------
    - One patient folder = one case
    - One dominant DICOM series per case
    - Pre-op scan is the only relevant scan present or first valid one
    """
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = [d for d in parent_dir.iterdir() if d.is_dir()]

    for site_path in sites:
        site = site_path.name
        print(f"\n=== Processing site: {site} ===")

        for patient_folder in site_path.iterdir():
            if not patient_folder.is_dir():
                continue

            # ----------------------------
            # Extract patient ID
            # ----------------------------
            patient_id = extract_patient_id(patient_folder.name)

            # ----------------------------
            # Locate DICOM data
            # ----------------------------
            dicom_dir = find_best_dicom_folder(patient_folder)

            if dicom_dir is None:
                print(f"No DICOM found: {patient_folder}")
                continue

            output_name = f"{patient_id}_{site}.nii.gz"
            output_path = output_dir / output_name

            if output_path.exists():
                continue

            try:
                img = convert_dicom_folder_to_nifti(str(dicom_dir))

                if img is None:
                    print(f"Invalid DICOM series: {dicom_dir}")
                    continue

                sitk.WriteImage(img, str(output_path))
                print(f"Saved: {output_name}")

            except Exception as e:
                print(f"Failed {patient_folder}: {e}")


# -------------------------
# RUN
# -------------------------
convert_dataset(
    parent_dir="/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/data/d",
    output_dir="/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr"
)