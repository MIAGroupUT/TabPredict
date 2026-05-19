"""
Tests for src/data/loaders.py

Covers:
- _clean_clinical_features: type coercion, None/empty → 0.0
- _load_aaa_metrics: valid JSON, missing file fallback
- load_splits_and_config: file reading, feature_names ordering, None cleaning
- compute_clin_stats: mean/std correctness
- build_data_list: sample structure, with/without complications
- BuildCTAd: 4-channel concatenation logic
- NormalizeClinicald: z-normalisation
- RuntimeDataset: __len__, __getitem__, transform application
- get_val_runtime_transforms: returns a Compose
- _tensor_keys: with/without complications
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from tests.conftest import CLINICAL_DIM, AAA_DIM


# ── _clean_clinical_features ──────────────────────────────────────────────────


def test_clean_string_numeric():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": "3.14"})
    assert cleaned["a"] == pytest.approx(3.14)


def test_clean_none_becomes_zero():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": None})
    assert cleaned["a"] == 0.0


def test_clean_empty_string_becomes_zero():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": ""})
    assert cleaned["a"] == 0.0


def test_clean_invalid_string_becomes_zero():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": "not_a_number"})
    assert cleaned["a"] == 0.0


def test_clean_float_passthrough():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": 2.5})
    assert cleaned["a"] == 2.5


def test_clean_preserves_all_keys():
    from src.data.loaders import _clean_clinical_features
    d = {"x": 1, "y": None, "z": ""}
    cleaned = _clean_clinical_features(d)
    assert set(cleaned.keys()) == {"x", "y", "z"}


# ── _load_aaa_metrics ─────────────────────────────────────────────────────────


def test_load_aaa_metrics_missing_file(tmp_path):
    from src.data.loaders import _load_aaa_metrics
    result = _load_aaa_metrics(tmp_path / "nonexistent.json")
    assert result == [0.0] * 5


def test_load_aaa_metrics_valid_json(tmp_path):
    from src.data.loaders import _load_aaa_metrics
    metrics = {
        "volume_lumen_ml": 10.0,
        "volume_ilt_ml":   5.0,
        "volume_total_ml": 15.0,
        "max_lumen_to_wall_dist_mm": 2.0,
        "avg_lumen_to_wall_dist_mm": 1.5,
    }
    p = tmp_path / "patient.json"
    p.write_text(json.dumps(metrics))
    result = _load_aaa_metrics(p)
    assert result == [10.0, 5.0, 15.0, 2.0, 1.5]


def test_load_aaa_metrics_partial_json(tmp_path):
    """Missing keys default to 0.0."""
    from src.data.loaders import _load_aaa_metrics
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"volume_lumen_ml": 7.0}))
    result = _load_aaa_metrics(p)
    assert result[0] == 7.0
    assert result[1] == 0.0   # volume_ilt_ml missing


def test_load_aaa_metrics_returns_five_values(tmp_path):
    from src.data.loaders import _load_aaa_metrics
    p = tmp_path / "full.json"
    p.write_text(json.dumps({"volume_lumen_ml": 1.0}))
    result = _load_aaa_metrics(p)
    assert len(result) == 5


# ── load_splits_and_config ────────────────────────────────────────────────────


def _make_fake_base_dir(tmp_path, pids=("p1", "p2"), n_features=3):
    feat_keys = [f"feat_{i}" for i in range(n_features)]
    splits = {"train": list(pids[:1]), "val": list(pids[1:2]), "test": []}
    config = {
        pid: {
            "pre_op_features": {k: float(i) for i, k in enumerate(feat_keys)},
            "ground_truth":     {"reintervention": idx % 2},
        }
        for idx, pid in enumerate(pids)
    }
    clin_dir = tmp_path / "clinical_data"
    clin_dir.mkdir()
    (tmp_path / "dataset_splits.json").write_text(json.dumps(splits))
    (clin_dir / "radar_multimodal_config_v1.json").write_text(json.dumps(config))
    return splits, config, feat_keys


def test_load_splits_returns_correct_splits(tmp_path):
    from src.data.loaders import load_splits_and_config
    expected_splits, _, _ = _make_fake_base_dir(tmp_path)
    splits, _, _ = load_splits_and_config(tmp_path)
    assert splits["train"] == expected_splits["train"]
    assert splits["val"]   == expected_splits["val"]


def test_load_splits_feature_names_sorted(tmp_path):
    from src.data.loaders import load_splits_and_config
    _make_fake_base_dir(tmp_path, n_features=5)
    _, _, feature_names = load_splits_and_config(tmp_path)
    assert feature_names == sorted(feature_names)


def test_load_splits_feature_count(tmp_path):
    from src.data.loaders import load_splits_and_config
    _make_fake_base_dir(tmp_path, n_features=4)
    _, _, feature_names = load_splits_and_config(tmp_path)
    assert len(feature_names) == 4


def test_load_splits_cleans_none_features(tmp_path):
    """None values in pre_op_features should become 0.0."""
    from src.data.loaders import load_splits_and_config
    config = {
        "p1": {
            "pre_op_features": {"a": None, "b": "bad_val"},
            "ground_truth": {"reintervention": 0},
        }
    }
    splits = {"train": ["p1"], "val": [], "test": []}
    clin_dir = tmp_path / "clinical_data"
    clin_dir.mkdir()
    (tmp_path / "dataset_splits.json").write_text(json.dumps(splits))
    (clin_dir / "radar_multimodal_config_v1.json").write_text(json.dumps(config))

    _, result_config, _ = load_splits_and_config(tmp_path)
    assert result_config["p1"]["pre_op_features"]["a"] == 0.0
    assert result_config["p1"]["pre_op_features"]["b"] == 0.0


# ── compute_clin_stats ────────────────────────────────────────────────────────


def test_compute_clin_stats_mean():
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 1.0, "b": 3.0}},
        "p2": {"pre_op_features": {"a": 3.0, "b": 5.0}},
    }
    mean, _ = compute_clin_stats(["p1", "p2"], config, ["a", "b"])
    np.testing.assert_allclose(mean, [2.0, 4.0])


def test_compute_clin_stats_std_positive():
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 0.0}},
        "p2": {"pre_op_features": {"a": 1.0}},
    }
    _, std = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert std[0] > 0  # +1e-8 floor


def test_compute_clin_stats_constant_feature():
    """Constant feature: std = 1e-8 (just the epsilon)."""
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 5.0}},
        "p2": {"pre_op_features": {"a": 5.0}},
    }
    mean, std = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert mean[0] == pytest.approx(5.0)
    assert std[0]  == pytest.approx(1e-8)


def test_compute_clin_stats_missing_key_defaults_zero():
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {}},  # missing "a"
        "p2": {"pre_op_features": {"a": 4.0}},
    }
    mean, _ = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert mean[0] == pytest.approx(2.0)  # (0 + 4) / 2


# ── build_data_list ───────────────────────────────────────────────────────────


def test_build_data_list_basic(tmp_path):
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {"a": 1.0}, "ground_truth": {"reintervention": 0}}}
    result = build_data_list(["p1"], tmp_path, config, ["a"])
    assert len(result) == 1
    item = result[0]
    assert item["pid"] == "p1"
    assert "clin"               in item
    assert "aaa_metrics"        in item
    assert "reintervention_label" in item


def test_build_data_list_clin_shape(tmp_path):
    from src.data.loaders import build_data_list
    feat_keys = ["a", "b", "c"]
    config = {"p1": {"pre_op_features": {"a": 1.0, "b": 2.0, "c": 3.0},
                     "ground_truth": {"reintervention": 1}}}
    result = build_data_list(["p1"], tmp_path, config, feat_keys)
    assert result[0]["clin"].shape == (3,)


def test_build_data_list_aaa_defaults(tmp_path):
    """Metrics file absent → all-zeros list of length 5."""
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {}, "ground_truth": {"reintervention": 0}}}
    result = build_data_list(["p1"], tmp_path, config, [])
    assert result[0]["aaa_metrics"] == [0.0] * 5


def test_build_data_list_reint_label(tmp_path):
    from src.data.loaders import build_data_list
    config = {
        "p1": {"pre_op_features": {}, "ground_truth": {"reintervention": 1}},
        "p2": {"pre_op_features": {}, "ground_truth": {"reintervention": 0}},
    }
    result = build_data_list(["p1", "p2"], tmp_path, config, [])
    assert result[0]["reintervention_label"][0] == 1.0
    assert result[1]["reintervention_label"][0] == 0.0


def test_build_data_list_with_complications(tmp_path):
    from src.data.loaders import build_data_list
    comps  = ["endoleak_1a", "migration"]
    config = {"p1": {"pre_op_features": {}, "ground_truth":
                     {"reintervention": 1, "endoleak_1a": 1}}}
    result = build_data_list(["p1"], tmp_path, config, [], complications=comps)
    assert "complication_labels" in result[0]
    labels = result[0]["complication_labels"]
    assert len(labels) == 2
    assert labels[0] == 1.0   # endoleak_1a present
    assert labels[1] == -1.0  # migration not in ground_truth → -1


def test_build_data_list_without_complications_no_comp_key(tmp_path):
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {}, "ground_truth": {"reintervention": 0}}}
    result = build_data_list(["p1"], tmp_path, config, [])
    assert "complication_labels" not in result[0]


# ── BuildCTAd ─────────────────────────────────────────────────────────────────


def test_build_ctad_output_key():
    from src.data.loaders import BuildCTAd
    ct  = torch.rand(1, 6, 6, 6)
    lbl = torch.zeros(1, 6, 6, 6)
    dst = torch.rand(1, 6, 6, 6)
    out = BuildCTAd()({"ct": ct, "lbl": lbl, "dst": dst})
    assert "cta" in out


def test_build_ctad_output_shape():
    from src.data.loaders import BuildCTAd
    ct  = torch.rand(1, 6, 6, 6)
    lbl = torch.zeros(1, 6, 6, 6)
    dst = torch.rand(1, 6, 6, 6)
    out = BuildCTAd()({"ct": ct, "lbl": lbl, "dst": dst})
    assert out["cta"].shape == (4, 6, 6, 6)


def test_build_ctad_channel_0_is_ct():
    from src.data.loaders import BuildCTAd
    ct  = torch.rand(1, 4, 4, 4)
    lbl = torch.zeros(1, 4, 4, 4)
    dst = torch.zeros(1, 4, 4, 4)
    out = BuildCTAd()({"ct": ct, "lbl": lbl, "dst": dst})
    assert torch.allclose(out["cta"][0], ct[0])


def test_build_ctad_channel_1_is_lumen():
    from src.data.loaders import BuildCTAd
    ct  = torch.zeros(1, 4, 4, 4)
    lbl = torch.zeros(1, 4, 4, 4)
    lbl[0, 1:3, 1:3, 1:3] = 1.0  # lumen region
    lbl[0, 0:1, 0:1, 0:1] = 2.0  # ILT region
    dst = torch.zeros(1, 4, 4, 4)
    out = BuildCTAd()({"ct": ct, "lbl": lbl, "dst": dst})
    expected_lumen = (lbl[0] == 1).float()
    assert torch.allclose(out["cta"][1], expected_lumen)


def test_build_ctad_channel_2_is_ilt():
    from src.data.loaders import BuildCTAd
    ct  = torch.zeros(1, 4, 4, 4)
    lbl = torch.zeros(1, 4, 4, 4)
    lbl[0, 0:1, 0:1, 0:1] = 2.0  # ILT
    dst = torch.zeros(1, 4, 4, 4)
    out = BuildCTAd()({"ct": ct, "lbl": lbl, "dst": dst})
    expected_ilt = (lbl[0] == 2).float()
    assert torch.allclose(out["cta"][2], expected_ilt)


def test_build_ctad_channel_3_is_dist():
    from src.data.loaders import BuildCTAd
    ct  = torch.zeros(1, 4, 4, 4)
    lbl = torch.zeros(1, 4, 4, 4)
    dst = torch.rand(1, 4, 4, 4)
    out = BuildCTAd()({"ct": ct, "lbl": lbl, "dst": dst})
    assert torch.allclose(out["cta"][3], dst[0])


# ── NormalizeClinicald ────────────────────────────────────────────────────────


def test_normalize_clinicald_correct_values():
    from src.data.loaders import NormalizeClinicald
    mean = np.array([1.0, 2.0], dtype=np.float32)
    std  = np.array([1.0, 2.0], dtype=np.float32)
    t    = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    data = {"clin": np.array([3.0, 6.0], dtype=np.float32)}
    out  = t(data)
    # (3-1)/(1+1e-8) ≈ 2.0,  (6-2)/(2+1e-8) ≈ 2.0
    assert torch.allclose(out["clin"], torch.tensor([2.0, 2.0]), atol=1e-4)


def test_normalize_clinicald_output_is_tensor():
    from src.data.loaders import NormalizeClinicald
    mean = np.zeros(3, dtype=np.float32)
    std  = np.ones(3, dtype=np.float32)
    t    = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    out  = t({"clin": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    assert isinstance(out["clin"], torch.Tensor)


def test_normalize_clinicald_zero_std_safe():
    """std=0 is padded with eps=1e-8, so no division by zero."""
    from src.data.loaders import NormalizeClinicald
    mean = np.array([5.0], dtype=np.float32)
    std  = np.array([0.0], dtype=np.float32)
    t    = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    out  = t({"clin": np.array([5.0], dtype=np.float32)})
    assert torch.isfinite(out["clin"]).all()


# ── RuntimeDataset ────────────────────────────────────────────────────────────


class _MockBase:
    def __init__(self, n):
        self._data = [{"val": torch.tensor([float(i)])} for i in range(n)]
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]


def test_runtime_dataset_len():
    from src.data.loaders import RuntimeDataset
    ds = RuntimeDataset(_MockBase(5), lambda d: d)
    assert len(ds) == 5


def test_runtime_dataset_getitem_applies_transform():
    from src.data.loaders import RuntimeDataset
    ds = RuntimeDataset(_MockBase(4), lambda d: {**d, "val": d["val"] * 10})
    item = ds[2]
    assert item["val"].item() == pytest.approx(20.0)


def test_runtime_dataset_identity_transform():
    from src.data.loaders import RuntimeDataset
    ds = RuntimeDataset(_MockBase(3), lambda d: d)
    assert ds[0]["val"].item() == pytest.approx(0.0)
    assert ds[1]["val"].item() == pytest.approx(1.0)


# ── transform constructors ────────────────────────────────────────────────────


def test_get_val_runtime_transforms_returns_callable():
    from src.data.loaders import get_val_runtime_transforms
    mean = np.zeros(CLINICAL_DIM, dtype=np.float32)
    std  = np.ones(CLINICAL_DIM, dtype=np.float32)
    tfm  = get_val_runtime_transforms(mean, std)
    assert callable(tfm)


def test_get_train_runtime_transforms_returns_callable():
    from src.data.loaders import get_train_runtime_transforms
    mean = np.zeros(CLINICAL_DIM, dtype=np.float32)
    std  = np.ones(CLINICAL_DIM, dtype=np.float32)
    tfm  = get_train_runtime_transforms(mean, std)
    assert callable(tfm)


def test_tensor_keys_without_complications():
    from src.data.loaders import _tensor_keys
    keys = _tensor_keys(False)
    assert "complication_labels" not in keys
    assert "reintervention_label" in keys


def test_tensor_keys_with_complications():
    from src.data.loaders import _tensor_keys
    keys = _tensor_keys(True)
    assert "complication_labels" in keys


# ── build_loaders ─────────────────────────────────────────────────────────────


def _build_loaders_args(tmp_path):
    """Minimal args for build_loaders: 4 train patients (2 pos / 2 neg) + 2 val."""
    pids_train = ["T000", "T001", "T002", "T003"]
    pids_val   = ["V000", "V001"]
    splits = {"train": pids_train, "val": pids_val, "test": []}
    config = {
        pid: {
            "pre_op_features": {"feat_a": 1.0, "feat_b": 0.5},
            "ground_truth": {"reintervention": i % 2},
        }
        for i, pid in enumerate(pids_train + pids_val)
    }
    feature_names = ["feat_a", "feat_b"]
    clin_mean = np.zeros(2, dtype=np.float32)
    clin_std  = np.ones(2,  dtype=np.float32)
    return splits, config, feature_names, clin_mean, clin_std


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_returns_two_dataloaders(mock_cache, mock_dl, tmp_path):
    """build_loaders must return a (train_loader, val_loader) tuple."""
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 2)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    result = build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, cache_dir=tmp_path / "cache", batch_size=2,
    )
    assert isinstance(result, tuple) and len(result) == 2


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_train_uses_weighted_sampler(mock_cache, mock_dl, tmp_path):
    """Train DataLoader must be built with a WeightedRandomSampler for class balance."""
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 4)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, cache_dir=tmp_path / "cache", batch_size=2,
    )
    train_kwargs = mock_dl.call_args_list[0].kwargs
    assert "sampler" in train_kwargs
    assert isinstance(train_kwargs["sampler"], WeightedRandomSampler)


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_val_uses_shuffle_false(mock_cache, mock_dl, tmp_path):
    """Val DataLoader must be built with shuffle=False (deterministic evaluation order)."""
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 2)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, cache_dir=tmp_path / "cache", batch_size=2,
    )
    val_kwargs = mock_dl.call_args_list[1].kwargs
    assert val_kwargs.get("shuffle") is False


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_batch_size_forwarded(mock_cache, mock_dl, tmp_path):
    """The requested batch_size must be forwarded to both DataLoaders."""
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 2)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, cache_dir=tmp_path / "cache", batch_size=4,
    )
    assert mock_dl.call_args_list[0].kwargs["batch_size"] == 4
    assert mock_dl.call_args_list[1].kwargs["batch_size"] == 4
