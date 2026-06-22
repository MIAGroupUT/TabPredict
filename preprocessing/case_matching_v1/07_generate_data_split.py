"""
06_generate_data_split.py — Patient-level stratified train/val/test split
==================================================================================

Splits at PATIENT level (not scan level), stratified by site + reintervention label.

A patient with pre-op + 3 post-op CTs counts as ONE unit for splitting. All their
scans end up in the same set so there is no data leakage across splits.

The split JSON records patient keys (e.g. "2637442_AUMC").
The DataLoader must expand these to individual scan NIfTIs at training time.

Output: training_subset_v1/dataset_splits_multitab.json
{
  "train":    ["2637442_AUMC", "387_UMCG", ...],
  "val":      [...],
  "test":     [...],
  "metadata": { "seed": 3299, "ratios": {...}, "total_patients": N }
}
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def generate_split(
    base_dir: str,
    config_json: str,
    seed: int = 3299,
    ratios: tuple = (0.7, 0.1, 0.2),
) -> None:
    """
    Generate a doubly-stratified (site × reintervention label) split.

    Only patients that have at least a pre-op CT NIfTI on disk are included.

    Parameters
    ----------
    base_dir : str
        training_subset_v1 root (imagesTr/ expected here).
    config_json : str
        Path to radar_multimodal_config_v1_multitab.json.
    seed : int
        Random seed for determinism.
    ratios : tuple
        (train, val, test) fractions — must sum to 1.
    """
    base_path  = Path(base_dir)
    images_dir = base_path / "imagesTr"
    output_path = base_path / "dataset_splits_multitab.json"

    with open(config_json) as f:
        config = json.load(f)

    strata: dict[str, list] = defaultdict(list)
    label_map: dict[str, int] = {}
    skipped_no_preop = 0

    for patient_key, info in config.items():
        preop = info["scans"].get("pre_op")
        if preop is None or preop.get("nifti") is None:
            skipped_no_preop += 1
            continue

        # Verify the pre-op NIfTI actually exists on disk
        if not Path(preop["nifti"]).exists():
            skipped_no_preop += 1
            continue

        site   = info["metadata"]["site_short"]
        reint  = 1 if info["ground_truth"]["reintervention"] else 0
        label_map[patient_key] = reint
        strata[f"{site}_{reint}"].append(patient_key)

    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []

    for stratum_key, ids in strata.items():
        ids = ids.copy()
        rng.shuffle(ids)
        n       = len(ids)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    splits = {
        "train": sorted(train_ids),
        "val":   sorted(val_ids),
        "test":  sorted(test_ids),
        "metadata": {
            "seed":            seed,
            "ratios":          {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "total_patients":  len(train_ids) + len(val_ids) + len(test_ids),
            "skipped_no_preop": skipped_no_preop,
        },
    }

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    # ---- Report ----
    print("\n" + "=" * 85)
    print(f" V1 PATIENT-LEVEL SPLIT | Seed: {seed}")
    print("=" * 85)
    print(f"{'Split':<8} | {'Patients':<8} | {'Pos':<5} | {'Neg':<5} | {'Prev':>7} | Site distribution")
    print("-" * 85)

    for name, ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        n   = len(ids)
        pos = sum(label_map[i] for i in ids)
        neg = n - pos
        prev = (pos / n * 100) if n else 0.0
        per_site: dict[str, int] = defaultdict(int)
        for pid in ids:
            per_site[pid.split("_")[-1]] += 1
        site_str = ", ".join(f"{s}: {c}" for s, c in sorted(per_site.items()))
        print(f"{name:<8} | {n:<8} | {pos:<5} | {neg:<5} | {prev:>6.1f}%  | {site_str}")

    print("-" * 85)
    tr_pos = sum(label_map[i] for i in train_ids)
    tr_neg = len(train_ids) - tr_pos
    pos_w  = tr_neg / tr_pos if tr_pos else 1.0
    print(f"Recommended train pos_weight: {pos_w:.3f}  ({tr_neg}/{tr_pos})")
    print(f"Patients skipped (no pre-op NIfTI): {skipped_no_preop}")
    print("=" * 85)
    print(f"Saved → {output_path}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    BASE_DIR    = "/deepstore/datasets/mia/AI4EVAR/derivatives/training_subset_v1"
    CONFIG_JSON = f"{BASE_DIR}/clinical_data/radar_multimodal_config_v1_multitab.json"

    generate_split(BASE_DIR, CONFIG_JSON, seed=3299, ratios=(0.7, 0.1, 0.2))
