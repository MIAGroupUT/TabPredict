import json
import random
from pathlib import Path
from collections import defaultdict

def generate_doubly_stratified_split_from_json(base_dir, json_path, seed=3299, ratios=(0.7, 0.1, 0.2)):
    """
    Generate deterministic STRATIFIED train/val/test split per SITE and per LABEL
    using the ground_truth.reintervention field from the config JSON.
    """
    base_path = Path(base_dir)
    image_dir = base_path / "imagesTr"
    output_split_path = base_path / "dataset_splits.json"

    # 1. Load the multimodal config JSON
    with open(json_path, 'r') as f:
        config_data = json.load(f)

    # 2. Group cases by Site + Label (The "Double Stratification" step)
    strata_groups = defaultdict(list)
    label_map = {} # To keep track for reporting

    # Iterate through the keys in your JSON (which should match your case IDs)
    for case_id, info in config_data.items():
        # Verify the image exists in imagesTr
        img_path = image_dir / f"{case_id}.nii.gz"
        if not img_path.exists():
            continue

        # Extract site and ground truth reintervention
        site = info['metadata']['site_short']
        # Convert boolean True/False to 1/0
        reint_label = 1 if info['ground_truth']['reintervention'] else 0
        
        label_map[case_id] = reint_label
        
        # Create the Stratum Key
        strata_key = f"{site}_{reint_label}"
        strata_groups[strata_key].append(case_id)

    if not strata_groups:
        raise RuntimeError(f"No matching images found in {image_dir} based on JSON keys.")

    # 3. Deterministic split per Stratum
    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []

    for strata_key, ids in strata_groups.items():
        ids = ids.copy()
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    # 4. Save splits
    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "metadata": {
            "seed": seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "total_cases": len(train_ids) + len(val_ids) + len(test_ids),
        }
    }

    with open(output_split_path, "w") as f:
        json.dump(splits, f, indent=4)

    # 5. Reporting (Exact Distribution Table)
    print("\n" + "=" * 85)
    print(f" DOUBLE STRATIFIED SPLIT REPORT (JSON-BASED) | Seed: {seed}")
    print("=" * 85)
    print(f"{'Split':<8} | {'Total':<5} | {'Pos (1)':<7} | {'Neg (0)':<7} | {'Prevalence':<10} | {'Site Distribution'}")
    print("-" * 85)

    for name, split_ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        n = len(split_ids)
        pos = sum(label_map[cid] for cid in split_ids)
        neg = n - pos
        prev = (pos / n) * 100 if n > 0 else 0
        
        per_site = defaultdict(int)
        for cid in split_ids:
            site = cid.split("_")[1] # Assumes ID is {pt}_{site}
            per_site[site] += 1

        site_str = ", ".join([f"{s}: {c}" for s, c in sorted(per_site.items())])
        print(f"{name:8} | {n:<5} | {pos:<7} | {neg:<7} | {prev:>8.2f}%  | {site_str}")

    print("-" * 85)
    # Calculate exact pos_weight for the training set
    train_pos = sum(label_map[i] for i in train_ids)
    train_neg = len(train_ids) - train_pos
    train_pos_weight = train_neg / train_pos if train_pos > 0 else 1.0
    
    print(f"RECOMMENDED TRAIN POS_WEIGHT: {train_pos_weight:.3f} ({train_neg}/{train_pos})")
    print("=" * 85)
    print(f"Saved → {output_split_path}")

# --- RUN ---
BASE_DIR = "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1"
JSON_PATH = f"{BASE_DIR}/clinical_data/radar_multimodal_config_v1.json"

generate_doubly_stratified_split_from_json(BASE_DIR, JSON_PATH, seed=3299, ratios=(0.7, 0.1, 0.2))