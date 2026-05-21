"""
04_data_organizing.py — Organise all NIfTIs into training_subset_v2/ (v2 pipeline)
==========================================================================================

Reads radar_multimodal_config_v2.json and copies the converted NIfTIs from
imagesTr_all/ into the structured training_subset_v2/ directories, using the
standardised file stems defined in the config.

Handles all modalities:
  - pre-op CT  → training_subset_v2/imagesTr/{spss_id}_{site}_s001_CT.nii.gz
  - post-op CTs → training_subset_v2/imagesTr/{spss_id}_{site}_s002_CT.nii.gz, …
  - US scans   → training_subset_v2/imagesTr/{spss_id}_{site}_s001_US.nii.gz, …
  - MR scans   → training_subset_v2/imagesTr/{spss_id}_{site}_s001_MR.nii.gz, …
  - XR scans   → training_subset_v2/imagesTr/{spss_id}_{site}_s001_XR.nii.gz, …

Source naming (output of 00_dicom_to_nifti_renaming.py)
-------------------------------------------------
imagesTr_all/{folder_id}_{site_abbr}_{session_id}.nii.gz
"""

import json
import shutil
from pathlib import Path


def scan_entries(patient_record: dict) -> list[dict]:
    """Return all scan entries for a patient: pre_op CT, pre_op_other, and post_op."""
    scans = patient_record["scans"]
    entries = []
    if scans.get("pre_op") is not None:
        entries.append(scans["pre_op"])
    for s in scans.get("pre_op_other", []):
        entries.append(s)
    for s in scans.get("post_op", []):
        entries.append(s)
    return entries


def organize_v2_dataset(config_json: str, nifti_source_dir: str) -> None:
    """
    Copy NIfTIs from imagesTr_all/ into training_subset_v2/imagesTr/.

    Parameters
    ----------
    config_json : str
        Path to radar_multimodal_config_v2.json.
    nifti_source_dir : str
        Flat directory produced by 00_dicom_to_nifti_v2.py
        (contains {folder_id}_{site}_{session_id}.nii.gz files).
    """
    with open(config_json) as f:
        config = json.load(f)

    source = Path(nifti_source_dir)
    copied = skipped_missing = already_done = 0

    for patient_key, patient_info in config.items():
        folder_id = patient_info["metadata"]["folder_id"]
        site_abbr = patient_info["metadata"]["site_short"]

        for scan in scan_entries(patient_info):
            nifti_path = scan.get("nifti")
            if nifti_path is None:
                continue

            dst = Path(nifti_path)
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                already_done += 1
                continue

            # Source: {folder_id}_{site}_{session_id}.nii.gz
            session_id = scan["session_id"]
            src_name   = f"{folder_id}_{site_abbr}_{session_id}.nii.gz"
            src        = source / src_name

            if not src.exists():
                print(f"[-] Missing source NIfTI: {src_name}  (patient {patient_key})")
                skipped_missing += 1
                continue

            shutil.copy2(src, dst)
            mod = scan["modality"]
            idx = scan["modality_index"]
            print(f"[+] {patient_key} | {mod} s{idx:03d} → {dst.name}")
            copied += 1

    print(f"\n{'='*60}")
    print(f"ORGANISE COMPLETE")
    print(f"  Copied           : {copied}")
    print(f"  Already present  : {already_done}")
    print(f"  Missing source   : {skipped_missing}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    CONFIG_JSON = (
        "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/"
        "clinical_data/radar_multimodal_config_v2.json"
    )
    NIFTI_SOURCE = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr_all"

    organize_v2_dataset(CONFIG_JSON, NIFTI_SOURCE)
