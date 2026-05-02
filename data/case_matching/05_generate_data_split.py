import json
import random
from pathlib import Path
from collections import defaultdict


def parse_case_id(filename: str):
    """
    Extract (patient_id, site_abbr) from filename.

    Expected format:
        0695_RNST.nii.gz

    Returns
    -------
    tuple[str, str]
        (patient_id, site_abbr)
    """
    stem = filename.replace(".nii.gz", "").replace(".nii", "")
    return stem.rsplit("_", 1)


def generate_stratified_split(base_dir, seed=3299, ratios=(0.7, 0.1, 0.2)):
    """
    Generate deterministic STRATIFIED train/val/test split per site.

    This ensures:
    - Each hospital/site is proportionally represented in all splits
    - Reproducibility via fixed seed
    - No site leakage or imbalance

    Parameters
    ----------
    base_dir : str or Path
        Path to training_subset_v1 folder.

    seed : int
        Random seed for reproducibility.

    ratios : tuple
        (train, val, test) split ratios.

    Output
    ------
    dataset_splits.json
        {
            "train": [...],
            "val": [...],
            "test": [...],
            "metadata": {...}
        }

    File format
    -----------
    IDs are stored as:
        {patient_id}_{site_abbr}
    """
    base_path = Path(base_dir)
    image_dir = base_path / "imagesTr"
    output_split_path = base_path / "dataset_splits.json"

    # -----------------------------------------------------
    # 1. Collect cases grouped by site
    # -----------------------------------------------------
    site_groups = defaultdict(list)

    for f in image_dir.glob("*.nii.gz"):
        patient_id, site = parse_case_id(f.name)
        case_id = f"{patient_id}_{site}"
        site_groups[site].append(case_id)

    if not site_groups:
        raise RuntimeError("No data found in imagesTr")

    # -----------------------------------------------------
    # 2. Deterministic stratified split per site
    # -----------------------------------------------------
    rng = random.Random(seed)

    train_ids, val_ids, test_ids = [], [], []

    for site, ids in site_groups.items():
        ids = ids.copy()
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    # -----------------------------------------------------
    # 3. Save splits
    # -----------------------------------------------------
    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "metadata": {
            "seed": seed,
            "ratios": {
                "train": ratios[0],
                "val": ratios[1],
                "test": ratios[2],
            },
            "total": len(train_ids) + len(val_ids) + len(test_ids),
            "sites": list(site_groups.keys())
        }
    }

    with open(output_split_path, "w") as f:
        json.dump(splits, f, indent=4)

    # -----------------------------------------------------
    # 4. Reporting
    # -----------------------------------------------------
    print("\n" + "=" * 50)
    print(f" STRATIFIED SPLIT (seed={seed})")
    print("=" * 50)

    for name, split in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        per_site = defaultdict(int)
        for case in split:
            site = case.split("_")[1]
            per_site[site] += 1

        print(f"{name:6} | n={len(split):4} | {dict(per_site)}")

    print("=" * 50)
    print(f"Saved → {output_split_path}")

# --- RUN ---
BASE_DIR = "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1"
generate_stratified_split(BASE_DIR, seed=3299, ratios=(0.7, 0.1, 0.2))