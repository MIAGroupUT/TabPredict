"""
Shared data-loading pipeline for both reintervention model variants.

Key entry points
────────────────
build_data_list()           Build a list of dicts for MONAI CacheDataset.
get_cached_transforms()     Expensive deterministic transforms (cached to RAM).
get_train_runtime_transforms()
get_val_runtime_transforms()
RuntimeDataset              Wraps a CacheDataset and applies per-access transforms.
load_splits_and_config()    Load dataset_splits.json + radar_multimodal_config.
compute_clin_stats()        Compute per-feature mean/std from training split.
"""

from __future__ import annotations

import json, random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

# default image geometry — can be overridden via config
DEFAULT_SIZE    = (224, 224, 160)
DEFAULT_SPACING = (1.5, 1.5, 3)
DEFAULT_HU_MIN  = -1000
DEFAULT_HU_MAX  =  1000

# Sentinel values used in RADAR to denote "not measured / not applicable".
# Treating these as real numbers corrupts z-normalisation statistics.
_SENTINEL_VALUES: frozenset = frozenset({999.0, 9999.0, 99999.0})

# Features that carry no clinical signal and must be dropped before training.
# PatientNr is a database ID (up to ~1e8) with no clinical meaning.
_EXCLUDE_FEATURES: frozenset = frozenset({"PatientNr"})

# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_splits_and_config(base_dir: Path) -> tuple[dict, dict, list[str]]:
    """
    Returns:
        splits       dict with "train"/"val"/"test" ID lists
        config       per-patient dict from radar_multimodal_config_v1.json
        feature_names sorted list of clinical feature keys
    """
    with open(base_dir / "dataset_splits.json") as f:
        splits = json.load(f)
    with open(base_dir / "clinical_data" / "radar_multimodal_config_v1.json") as f:
        config = json.load(f)

    first_pid = list(config.keys())[0]
    # Derive feature names BEFORE cleaning so we can filter excluded features.
    # _EXCLUDE_FEATURES (e.g. PatientNr) are dropped here and never seen by the model.
    feature_names = sorted(
        k for k in config[first_pid]["pre_op_features"].keys()
        if k not in _EXCLUDE_FEATURES
    )

    for pid in config:
        if "pre_op_features" in config[pid]:
            config[pid]["pre_op_features"] = _clean_clinical_features(
                config[pid]["pre_op_features"]
            )

    return splits, config, feature_names


def filter_features_by_missingness(
    train_ids: list[str],
    config: dict,
    feature_names: list[str],
    max_missing_pct: float = 1.0,
) -> list[str]:
    """Drop features that are missing in more than max_missing_pct of training patients.

    Missingness is computed on the training split only to avoid leaking val/test
    statistics into feature selection.  The returned list preserves the original
    sort order so downstream indexing stays consistent.
    """
    if max_missing_pct >= 1.0:
        return feature_names
    n = len(train_ids)
    kept = []
    for k in feature_names:
        n_missing = sum(
            1 for pid in train_ids
            if np.isnan(config.get(pid, {}).get("pre_op_features", {}).get(k, np.nan))
        )
        if n_missing / n < max_missing_pct:
            kept.append(k)
    return kept


def compute_clin_stats(
    train_ids: list[str],
    config: dict,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from the training split.

    NaN values (missing measurements, sentinel-replaced values) are excluded
    from the statistics so they do not corrupt the mean or std of real values.
    Features that are 100% missing receive mean=0, std=1 so they normalise
    to z=0 for every patient and the gate can cheaply zero them out.
    """
    matrix = np.stack([
        np.array(
            [config[pid]["pre_op_features"].get(k, np.nan) for k in feature_names],
            dtype=np.float32,
        )
        for pid in train_ids
    ])
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix,  axis=0)
    # All-NaN column (100% missing feature): nanmean/nanstd return NaN — set neutral values
    mean = np.where(np.isnan(mean), 0.0, mean)
    std  = np.where(np.isnan(std),  1.0, std)
    return mean.astype(np.float32), (std + 1e-8).astype(np.float32)


def _clean_clinical_features(feat_dict: dict) -> dict:
    """Convert raw RADAR feature values to float, replacing bad values with NaN.

    Handles three classes of problematic values:
    - Sentinel codes (999, 9999, 99999): RADAR uses these for "not measured".
      Treating them as real numbers corrupts z-normalisation statistics.
    - Non-numeric strings (date fields like "2010-12-07", free text): silently
      became 0.0 before; now become NaN.
    - Excluded features (PatientNr): database IDs with no clinical meaning.

    NaN values propagate to NormalizeClinicald which imputes them to z=0
    (the training-set mean), the most agnostic choice for missing data.
    """
    cleaned = {}
    for k, v in feat_dict.items():
        if k in _EXCLUDE_FEATURES:
            continue
        try:
            f = float(v) if v not in ("", None) else np.nan
            # Sentinel → NaN (do not use isnan check on f before membership test;
            # NaN is not in the frozenset so the order here is safe)
            if not np.isnan(f) and f in _SENTINEL_VALUES:
                f = np.nan
            cleaned[k] = f
        except (TypeError, ValueError):
            cleaned[k] = np.nan
    return cleaned


# ── dataset builder ──────────────────────────────────────────────────────────
def build_data_list(
    ids: list[str],
    base_dir: Path,
    config: dict,
    feature_keys: list[str],
    complications: list[str] | None = None,
) -> list[dict]:
    """
    Build a flat list of sample dicts ready for MONAI CacheDataset.

    Args:
        ids:           Patient ID list (e.g. splits["train"]).
        base_dir:      Root of training_subset_v1.
        config:        Per-patient config (from load_splits_and_config).
        feature_keys:  Ordered list of clinical feature names.
        complications: If provided, per-complication GT labels are included
                       (value -1 = missing/unknown).
    """
    with_comp = bool(complications)
    data = []
    for pid in ids:
        pdata      = config.get(pid, {})
        clin_dict  = pdata.get("pre_op_features", {})
        # Preserve NaN for missing / sentinel values — NormalizeClinicald imputes
        # them to z=0 (training mean) after normalisation, not before.
        clin = np.array(
            [clin_dict.get(k, np.nan) for k in feature_keys], dtype=np.float32
        )

        pre_op      = (pdata.get("scans", {}).get("pre_op")) or {}
        aaa_metrics = _load_aaa_metrics(pre_op.get("metrics"))
        gt          = pdata.get("ground_truth", {})
        reint_label = float(gt.get("reintervention", 0))

        sample: dict[str, Any] = {
            "ct":                   str(pre_op.get("nifti")),
            "lbl":                  str(pre_op.get("label")),
            "dst":                  str(pre_op.get("dist_map")),
            "clin":                 clin,
            "aaa_metrics":          aaa_metrics,
            "reintervention_label": np.array([reint_label], dtype=np.float32),
            "pid":                  pid,
        }
        if with_comp:
            sample["complication_labels"] = np.array(
                [float(gt.get(c, -1.0)) for c in complications], dtype=np.float32
            )
        data.append(sample)
    return data


def _load_aaa_metrics(path) -> list[float]:
    if path is None:
        return [0.0] * 5
    path = Path(path)
    if not path.exists():
        return [0.0] * 5
    with open(path) as f:
        m = json.load(f)
    return [
        m.get("volume_lumen_ml", 0.0),
        m.get("volume_ilt_ml", 0.0),
        m.get("volume_total_ml", 0.0),
        m.get("max_lumen_to_wall_dist_mm", 0.0),
        m.get("avg_lumen_to_wall_dist_mm", 0.0),
    ]


def build_tabular_arrays(
    ids: list[str],
    config: dict,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract tabular features and labels as raw numpy arrays.

    Returns:
        X     (N, len(feature_names) + 5) float32 — clinical features (raw, NaN for
              missing values) concatenated with 5 raw AAA geometry scalars
        y     (N,) float32 — reintervention binary labels
        pids  patient IDs in the same order as rows

    NaN values in X represent missing clinical measurements; callers are
    responsible for imputation (e.g. median or mean) before passing to a model.
    """
    X_rows, y_vals, pids = [], [], []
    for pid in ids:
        pdata     = config.get(pid, {})
        clin_dict = pdata.get("pre_op_features", {})
        clin      = np.array(
            [clin_dict.get(k, np.nan) for k in feature_names], dtype=np.float32
        )
        pre_op = (pdata.get("scans", {}).get("pre_op")) or {}
        aaa    = np.array(_load_aaa_metrics(pre_op.get("metrics")), dtype=np.float32)
        X_rows.append(np.concatenate([clin, aaa]))
        y_vals.append(float(pdata.get("ground_truth", {}).get("reintervention", 0)))
        pids.append(pid)
    return (
        np.stack(X_rows).astype(np.float32),
        np.array(y_vals, dtype=np.float32),
        pids,
    )


# ── MONAI transforms ─────────────────────────────────────────────────────────

class BuildCTAd(MapTransform):
    """Concatenate ct + lumen mask + ILT mask + distance map → 4-channel CTA."""

    def __init__(self, keys: list[str] = None) -> None:  # noqa: RUF013
        super().__init__(keys or ["ct", "lbl", "dst"])

    def __call__(self, data: dict) -> dict:
        ct, lbl, dst = data["ct"], data["lbl"], data["dst"]
        data["cta"] = torch.cat([ct, (lbl == 1).float(), (lbl == 2).float(), dst], dim=0)
        return data


class NormalizeClinicald(MapTransform):
    """Z-normalise clinical features using training-set statistics."""

    def __init__(
        self,
        keys: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(keys)
        self.mean = torch.tensor(mean)
        self.std  = torch.tensor(std)
        self.eps  = eps

    def __call__(self, data: dict) -> dict:
        clin = torch.as_tensor(data["clin"]).float()
        normalised = (clin - self.mean) / (self.std + self.eps)
        # NaN (missing or sentinel-replaced values) → 0.0 = training mean in z-space.
        # This is the most agnostic imputation: the feature contributes no signal
        # for this patient, so the gate can safely zero it out.
        data["clin"] = torch.nan_to_num(normalised, nan=0.0)
        return data


def get_cached_transforms(
    size: tuple[int, int, int] = DEFAULT_SIZE,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    hu_min: float = DEFAULT_HU_MIN,
    hu_max: float = DEFAULT_HU_MAX,
) -> Compose:
    """Expensive deterministic I/O + geometry — cached in RAM by CacheDataset."""
    return Compose([
        LoadImaged(keys=["ct", "lbl", "dst"]),
        EnsureChannelFirstd(keys=["ct", "lbl", "dst"]),
        Orientationd(keys=["ct", "lbl", "dst"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["ct", "lbl", "dst"],
            pixdim=spacing,
            mode=("bilinear", "nearest", "bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["ct"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["ct", "lbl", "dst"], spatial_size=size),
        CenterSpatialCropd(keys=["ct", "lbl", "dst"], roi_size=size),
    ])


def _tensor_keys(with_complications: bool) -> list[str]:
    keys = ["ct", "lbl", "dst", "aaa_metrics", "reintervention_label"]
    if with_complications:
        keys.append("complication_labels")
    return keys


_CT  = ["ct"]
_IMG = ["ct", "lbl", "dst"]


def get_train_runtime_transforms(
    clin_mean: np.ndarray,
    clin_std: np.ndarray,
    complications: list[str] | None = None,
) -> Compose:
    """Per-iteration augmentation + CTA build + normalisation.

    Spatial augmentations operate on ct+lbl+dst together (before BuildCTAd)
    so all three volumes stay geometrically consistent.  Intensity augmentations
    operate on ct only — lbl is a label map and dst is a distance field; neither
    should be blurred, shifted, or contrast-scaled.

    Note: aaa_metrics scalars are pre-computed from the original scan and are
    NOT recomputed here, so small spatial perturbations (zoom, affine) introduce
    a minor image/scalar inconsistency.  This is accepted: the scalars are the
    clinically authoritative values; the model learns robustness from the mismatch.
    """
    return Compose([
        # ── spatial (must precede BuildCTAd) ─────────────────────────────────
        RandFlipd(keys=_IMG, spatial_axis=0, prob=0.5),
        RandFlipd(keys=_IMG, spatial_axis=1, prob=0.5),
        RandFlipd(keys=_IMG, spatial_axis=2, prob=0.4),
        # 90° axial rotations — aggressive for aortic anatomy; monitor val AUC
        RandRotate90d(keys=_IMG, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandAffined(
            keys=_IMG,
            prob=0.3,
            rotate_range=(0.175, 0.175, 0.175),
            translate_range=(10, 10, 5),
            mode=("bilinear", "nearest", "bilinear"),
            padding_mode="border",
        ),
        RandZoomd(
            keys=_IMG,
            min_zoom=0.85, max_zoom=1.15,
            prob=0.3,
            mode=("trilinear", "nearest", "trilinear"),
            keep_size=True,
            padding_mode="edge",
        ),
        # ── intensity (ct only) ───────────────────────────────────────────────
        RandAdjustContrastd(keys=_CT, gamma=(0.7, 1.3), prob=0.3),
        RandShiftIntensityd(keys=_CT, offsets=0.1, prob=0.3),
        RandScaleIntensityd(keys=_CT, factors=0.15, prob=0.5),
        RandBiasFieldd(keys=_CT, prob=0.2, coeff_range=(0.0, 0.3)),
        RandGaussianNoised(keys=_CT, prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=_CT, prob=0.2,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.25, 0.5),
        ),
        # ── build 4-channel CTA tensor ────────────────────────────────────────
        BuildCTAd(),
        ToTensord(keys=_tensor_keys(bool(complications))),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


def get_val_runtime_transforms(
    clin_mean: np.ndarray,
    clin_std: np.ndarray,
    complications: list[str] | None = None,
) -> Compose:
    """Deterministic val/test pipeline — no augmentation."""
    return Compose([
        BuildCTAd(),
        ToTensord(keys=_tensor_keys(bool(complications))),
        NormalizeClinicald(keys=["clin"], mean=clin_mean, std=clin_std),
    ])


# ── dataset wrapper ──────────────────────────────────────────────────────────

class RuntimeDataset(torch.utils.data.Dataset):
    """
    Wraps a MONAI CacheDataset (slow, deterministic transforms, cached) and
    applies a runtime transform (fast, possibly stochastic) on every access.
    """

    def __init__(
        self,
        base_ds: CacheDataset,
        runtime_transform: Compose,
    ) -> None:
        self.base_ds = base_ds
        self.runtime = runtime_transform

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict:
        return self.runtime(self.base_ds[idx])


def _make_weighted_sampler(
    train_files: list[dict],
    seed: int = 3299,
) -> "torch.utils.data.WeightedRandomSampler":
    """Return a WeightedRandomSampler that draws positives and negatives equally.

    Each positive sample gets weight 1/N_pos and each negative gets 1/N_neg, so
    every training epoch contains an equal expected number of each class regardless
    of the natural ~18/82 split.  pos_weight in the loss can then stay at or near
    the raw class ratio without double-counting.
    """
    from torch.utils.data import WeightedRandomSampler

    labels  = [int(s["reintervention_label"][0]) for s in train_files]
    n_pos   = max(sum(labels), 1)
    n_neg   = max(len(labels) - n_pos, 1)
    w_pos   = 1.0 / n_pos
    w_neg   = 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]

    g = torch.Generator()
    g.manual_seed(seed)
    print(f"WeightedRandomSampler: {n_pos} pos / {n_neg} neg  "
          f"(w_pos={w_pos:.4f}, w_neg={w_neg:.4f})")
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                 replacement=True, generator=g)


def build_loaders(
    splits: dict,
    config: dict,
    feature_names: list[str],
    clin_mean: np.ndarray,
    clin_std: np.ndarray,
    base_dir: Path,
    batch_size: int = 8,
    num_workers: int = 16,
    complications: list[str] | None = None,
    train_cache_rate: float = 0.5,
    val_cache_rate: float = 1.0,
    size: tuple[int, int, int] = DEFAULT_SIZE,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    seed: int = 3299,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and val DataLoaders.

    The train loader uses WeightedRandomSampler to present ~50/50 positive/negative
    batches, eliminating the natural ~18/82 class skew.  pos_weight in the loss
    then encodes the residual clinical preference for recall over precision, without
    needing to be inflated to compensate for imbalance.
    """
    from torch.utils.data import DataLoader

    train_files = build_data_list(splits["train"], base_dir, config, feature_names, complications)
    val_files   = build_data_list(splits["val"],   base_dir, config, feature_names, complications)

    cached_tfm = get_cached_transforms(size=size, spacing=spacing)
    train_tfm  = get_train_runtime_transforms(clin_mean, clin_std, complications)
    val_tfm    = get_val_runtime_transforms(clin_mean, clin_std, complications)

    train_ds = RuntimeDataset(
        CacheDataset(data=train_files, transform=cached_tfm,
                     cache_rate=train_cache_rate, num_workers=num_workers // 2),
        train_tfm,
    )
    val_ds = RuntimeDataset(
        CacheDataset(data=val_files, transform=cached_tfm,
                     cache_rate=val_cache_rate, num_workers=num_workers // 2),
        val_tfm,
    )

    # Each DataLoader worker gets a deterministic seed derived from the base seed
    # so that per-worker augmentation (noise, intensity scale) is reproducible.
    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    sampler = _make_weighted_sampler(train_files, seed=seed)

    # Val is 100%-cached so it needs far fewer workers; fewer workers also reduces
    # the persistent-worker RSS accumulation that causes OOM on long runs.
    val_num_workers = max(2, num_workers // 4)

    train_kwargs = dict(num_workers=num_workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=1,
                        worker_init_fn=_worker_init)
    val_kwargs   = dict(num_workers=val_num_workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=1,
                        worker_init_fn=_worker_init)
    # shuffle=True is mutually exclusive with sampler; sampler handles ordering.
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, **train_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **val_kwargs)
    return train_loader, val_loader
