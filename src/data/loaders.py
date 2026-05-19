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
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandScaleIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

# default image geometry — can be overridden via config
DEFAULT_SIZE    = (128, 128, 128)
DEFAULT_SPACING = (1.5, 1.5, 1.5)
DEFAULT_HU_MIN  = -200
DEFAULT_HU_MAX  =  400


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
    feature_names = sorted(config[first_pid]["pre_op_features"].keys())

    for pid in config:
        if "pre_op_features" in config[pid]:
            config[pid]["pre_op_features"] = _clean_clinical_features(
                config[pid]["pre_op_features"]
            )

    return splits, config, feature_names


def compute_clin_stats(
    train_ids: list[str],
    config: dict,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from the training split."""
    matrix = np.stack([
        np.array(
            [config[pid]["pre_op_features"].get(k, 0.0) for k in feature_names],
            dtype=np.float32,
        )
        for pid in train_ids
    ])
    return matrix.mean(axis=0), matrix.std(axis=0) + 1e-8


def _clean_clinical_features(feat_dict: dict) -> dict:
    cleaned = {}
    for k, v in feat_dict.items():
        try:
            cleaned[k] = float(v) if v not in ("", None) else 0.0
        except (TypeError, ValueError):
            cleaned[k] = 0.0
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
        clin = np.nan_to_num(
            np.array([clin_dict.get(k, 0.0) for k in feature_keys], dtype=np.float32)
        )

        aaa_metrics = _load_aaa_metrics(base_dir / "metricsTr" / f"{pid}.json")
        gt          = pdata.get("ground_truth", {})
        reint_label = float(gt.get("reintervention", 0))

        sample: dict[str, Any] = {
            "ct":                   str(base_dir / "imagesTr"   / f"{pid}.nii.gz"),
            "lbl":                  str(base_dir / "labelsTr"   / f"{pid}.nii.gz"),
            "dst":                  str(base_dir / "distMapsTr" / f"{pid}.nii.gz"),
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


def _load_aaa_metrics(path: Path) -> list[float]:
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
        data["clin"] = (clin - self.mean) / (self.std + self.eps)
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
        Orientationd(keys=["ct", "lbl", "dst"], axcodes="RAS"),
        Spacingd(
            keys=["ct", "lbl", "dst"],
            pixdim=spacing,
            mode=("bilinear", "nearest", "bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["ct"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True
        ),
        Resized(
            keys=["ct", "lbl", "dst"],
            spatial_size=size,
            mode=("trilinear", "nearest", "trilinear"),
        ),
    ])


def _tensor_keys(with_complications: bool) -> list[str]:
    keys = ["ct", "lbl", "dst", "aaa_metrics", "reintervention_label"]
    if with_complications:
        keys.append("complication_labels")
    return keys


def get_train_runtime_transforms(
    clin_mean: np.ndarray,
    clin_std: np.ndarray,
    complications: list[str] | None = None,
) -> Compose:
    """Per-iteration augmentation + CTA build + normalisation."""
    return Compose([
        RandGaussianNoised(keys=["ct"], prob=0.3, mean=0.0, std=0.02),
        RandBiasFieldd(keys=["ct"], prob=0.2),
        RandScaleIntensityd(keys=["ct"], factors=0.1, prob=0.5),
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


def build_loaders(
    splits: dict,
    config: dict,
    feature_names: list[str],
    clin_mean: np.ndarray,
    clin_std: np.ndarray,
    base_dir: Path,
    cache_dir: Path,
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
    Build train and val DataLoaders with class-balanced sampling on
    the reintervention label.
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

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

    # Class-balanced sampler: ~50/50 positives in every batch
    train_labels   = [int(item["reintervention_label"][0]) for item in train_files]
    class_counts   = np.bincount(train_labels)
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), len(sample_weights),
        replacement=True, generator=sampler_generator,
    )

    # Each DataLoader worker gets a deterministic seed derived from the base seed
    # so that per-worker augmentation (noise, intensity scale) is reproducible.
    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4,
                         worker_init_fn=_worker_init)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,  **loader_kwargs)
    return train_loader, val_loader
