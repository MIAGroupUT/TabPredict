import json
import shutil
import os
from pathlib import Path


def organize_multimodal_dataset(
    config_json,
    source_img_dir,
    source_lbl_dir,
    source_cfd_dir,
    target_base_dir
):
    """
    Organize a multimodal dataset into a structured training directory.

    This function takes the output JSON from the 03_data_matching.py pipeline
    and copies all associated modalities into a clean training folder.

    Expected input structure (per patient):
    --------------------------------------
    matched_data[p_id] = {
        metadata: {
            img: filename,
            cfd: filename or None
        }
    }

    Output structure:
    -----------------
    training_subset_v1/
        imagesTr/
        labelsTr/
        cfdTr/

    Parameters
    ----------
    config_json : str
        Path to JSON file containing matched multimodal dataset.

    source_img_dir : str
        Directory containing CT images (.nii.gz).

    source_lbl_dir : str or None
        Directory containing labels (optional).

    source_cfd_dir : str
        Directory containing CFD files (.vtp). Used only if available.

    target_base_dir : str
        Base output directory where structured dataset will be created.

    Notes
    -----
    - CFD files are OPTIONAL (may be None in metadata).
    - Missing modalities are skipped with warnings.
    """

    with open(config_json, 'r') as f:
        matched_data = json.load(f)

    # -----------------------------------------------------
    # OUTPUT STRUCTURE
    # -----------------------------------------------------
    project_dir = Path(target_base_dir) / "training_subset_v1"

    t_img_dir = project_dir / "imagesTr"
    t_lbl_dir = project_dir / "labelsTr"
    t_cfd_dir = project_dir / "cfdTr"

    for d in [t_img_dir, t_lbl_dir, t_cfd_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Organizing {len(matched_data)} cases into: {project_dir}")

    copy_count = 0

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    for _, info in matched_data.items():
        p_id = info["metadata"]["id"]
        site_abbr = info["metadata"]["site_short"]
        print(f"Processing {p_id} from {site_abbr}...")
        new_filename = f"{p_id}_{site_abbr}.nii.gz"
        cfd_filename = f"{p_id}_{site_abbr}.vtp"

        # =====================================================
        # 1. COPY IMAGE
        # =====================================================
        src_img = Path(source_img_dir) / info['metadata']['img']

        if src_img.exists():
            shutil.copy2(src_img, t_img_dir / new_filename)
        else:
            print(f"[-] Missing image: {p_id}")

        # =====================================================
        # 2. COPY LABEL (optional)
        # =====================================================
        if source_lbl_dir:
            src_lbl = Path(source_lbl_dir) / info['metadata']['img']

            if src_lbl.exists():
                shutil.copy2(src_lbl, t_lbl_dir / new_filename)
            else:
                print(f"[-] Missing label: {p_id}")

        # =====================================================
        # 3. COPY CFD (OPTIONAL SAFE HANDLING)
        # =====================================================
        cfd_name = info['metadata'].get('cfd')

        if cfd_name is not None and source_cfd_dir is not None:
            src_cfd = Path(source_cfd_dir) / cfd_name

            if src_cfd.exists():
                shutil.copy2(src_cfd, t_cfd_dir / cfd_filename)
            else:
                print(f"[-] Missing CFD file on disk: {p_id}")
        else:
            # CFD is expected to be optional → no warning spam
            pass

        copy_count += 1

    print(f"Success: {copy_count} cases processed.")


# ---------------------------------------------------------
# CONFIGURATION PATHS
# ---------------------------------------------------------
CONFIG_PATH = "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1/clinical_data/radar_multimodal_config_v1.json"

SRC_IMG = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr_pre-op"
SRC_LBL = None
SRC_CFD = "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/renamed"

TARGET  = "/deepstore/datasets/mia/AI4EVAR/derivatives"


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
organize_multimodal_dataset(CONFIG_PATH, SRC_IMG, SRC_LBL, SRC_CFD, TARGET)