"""
04_data_organizing.py — Organise all NIfTIs into training_subset_v1/ (v1 pipeline)
==========================================================================================

Reads radar_multimodal_config_v1.json and copies the converted NIfTIs from
imagesTr_all/ into the structured training_subset_v1/ directories, using the
standardised file stems defined in the config.

Handles all modalities:
  - pre-op CT   → training_subset_v1/imagesTr/{spss_id}_{site}_s001_CT.nii.gz
  - post-op CTs → training_subset_v1/imagesTr/{spss_id}_{site}_s002_CT.nii.gz, …
  - US scans    → training_subset_v1/imagesTr/{spss_id}_{site}_s001_US.nii.gz, …
  - MR scans    → training_subset_v1/imagesTr/{spss_id}_{site}_s001_MR.nii.gz, …
  - XR scans    → training_subset_v1/imagesTr/{spss_id}_{site}_s001_XR.nii.gz, …

Source naming (output of 00_dicom_to_nifti_renaming.py)
-------------------------------------------------
imagesTr_all/{folder_id}_{site_abbr}_{session_id}.nii.gz
"""

import json
import shutil
from pathlib import Path


def scan_entries(patient_record: dict) -> list[dict]:
    """Return all scan entries for a patient across all timing buckets."""
    scans = patient_record["scans"]
    entries = []
    if scans.get("pre_op") is not None:
        entries.append(scans["pre_op"])
    for s in scans.get("pre_op_other", []):
        entries.append(s)
    for s in scans.get("post_op", []):
        entries.append(s)
    for s in scans.get("undated", []):
        entries.append(s)
    return entries


def organize_v1_dataset(
    config_json: str,
    nifti_source_dir: str,
    cfd_renamed_dir: str | None = None,
) -> None:
    """
    Copy NIfTIs (and optionally CFD meshes) into training_subset_v1/.

    Parameters
    ----------
    config_json : str
        Path to radar_multimodal_config_v1.json.
    nifti_source_dir : str
        Flat directory produced by 00_dicom_to_nifti_v1.py
        (contains {folder_id}_{site}_{session_id}.nii.gz files).
    cfd_renamed_dir : str | None
        Directory produced by 02_cfd_name_cleaning.py
        (contains {folder_id}_{site}_{session_id}.vtp files).
        Pass None to skip CFD copying.
    """
    with open(config_json) as f:
        config = json.load(f)

    source     = Path(nifti_source_dir)
    cfd_source = Path(cfd_renamed_dir) if cfd_renamed_dir else None

    copied = skipped_missing = already_done = 0
    cfd_copied = cfd_missing = cfd_done = 0

    for patient_key, patient_info in config.items():
        folder_id = patient_info["metadata"]["folder_id"]
        site_abbr = patient_info["metadata"]["site_short"]

        for scan in scan_entries(patient_info):
            session_id = scan["session_id"]

            # ---- NIfTI ----
            nifti_path = scan.get("nifti")
            if nifti_path is not None:
                dst = Path(nifti_path)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    already_done += 1
                else:
                    src_name = f"{folder_id}_{site_abbr}_{session_id}.nii.gz"
                    src      = source / src_name
                    if not src.exists():
                        print(f"[-] Missing NIfTI: {src_name}  (patient {patient_key})")
                        skipped_missing += 1
                    else:
                        shutil.copy2(src, dst)
                        mod = scan["modality"]
                        idx = scan["modality_index"]
                        print(f"[+] {patient_key} | {mod} s{idx:03d} → {dst.name}")
                        copied += 1

            # ---- CFD ----
            cfd_path = scan.get("cfd")
            if cfd_source is not None and cfd_path is not None:
                cfd_dst = Path(cfd_path)
                cfd_dst.parent.mkdir(parents=True, exist_ok=True)
                if cfd_dst.exists():
                    cfd_done += 1
                else:
                    cfd_src_name = f"{folder_id}_{site_abbr}_{session_id}.vtp"
                    cfd_src      = cfd_source / cfd_src_name
                    if not cfd_src.exists():
                        print(f"[-] Missing CFD: {cfd_src_name}  (patient {patient_key})")
                        cfd_missing += 1
                    else:
                        shutil.copy2(cfd_src, cfd_dst)
                        print(f"[+] {patient_key} | CFD → {cfd_dst.name}")
                        cfd_copied += 1

    print(f"\n{'='*60}")
    print(f"ORGANISE COMPLETE")
    print(f"  NIfTI copied     : {copied}")
    print(f"  NIfTI present    : {already_done}")
    print(f"  NIfTI missing    : {skipped_missing}")
    if cfd_source is not None:
        print(f"  CFD copied       : {cfd_copied}")
        print(f"  CFD present      : {cfd_done}")
        print(f"  CFD missing      : {cfd_missing}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    CONFIG_JSON = (
        "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v2/"
        "clinical_data/radar_multimodal_config_v1.json"
    )
    NIFTI_SOURCE = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr_all"
    CFD_RENAMED  = "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/renamed_v1"

    organize_v1_dataset(CONFIG_JSON, NIFTI_SOURCE, cfd_renamed_dir=CFD_RENAMED)
