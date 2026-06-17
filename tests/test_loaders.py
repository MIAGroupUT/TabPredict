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


def test_clean_none_becomes_nan():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": None})
    assert np.isnan(cleaned["a"])


def test_clean_empty_string_becomes_nan():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": ""})
    assert np.isnan(cleaned["a"])


def test_clean_invalid_string_becomes_nan():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": "not_a_number"})
    assert np.isnan(cleaned["a"])


def test_clean_float_passthrough():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": 2.5})
    assert cleaned["a"] == 2.5


def test_clean_preserves_all_keys():
    from src.data.loaders import _clean_clinical_features
    d = {"x": 1, "y": None, "z": ""}
    cleaned = _clean_clinical_features(d)
    assert set(cleaned.keys()) == {"x", "y", "z"}


# ── sentinel and exclusion handling ──────────────────────────────────────────


def test_clean_sentinel_9999_becomes_nan():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": 9999.0})
    assert np.isnan(cleaned["a"]), "9999 sentinel must become NaN"


def test_clean_sentinel_999_becomes_nan():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": 999.0})
    assert np.isnan(cleaned["a"]), "999 sentinel must become NaN"


def test_clean_sentinel_99999_becomes_nan():
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": 99999.0})
    assert np.isnan(cleaned["a"]), "99999 sentinel must become NaN"


def test_clean_sentinel_as_string_becomes_nan():
    """Sentinels may arrive as string '9999' from JSON."""
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": "9999"})
    assert np.isnan(cleaned["a"])


def test_clean_near_sentinel_is_not_sentinel():
    """9998 is a real measurement value, not a sentinel."""
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": 9998.0})
    assert cleaned["a"] == pytest.approx(9998.0)


def test_clean_date_string_becomes_nan():
    """Date fields like '2010-12-07' cannot be cast to float → NaN."""
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"a": "2010-12-07"})
    assert np.isnan(cleaned["a"])


def test_clean_excludes_patient_nr():
    """PatientNr (a database ID) must be dropped entirely."""
    from src.data.loaders import _clean_clinical_features
    cleaned = _clean_clinical_features({"PatientNr": "12345", "age": 72.0})
    assert "PatientNr" not in cleaned
    assert cleaned["age"] == pytest.approx(72.0)


def test_feature_names_excludes_patient_nr(tmp_path):
    """load_splits_and_config must not include PatientNr in feature_names."""
    from src.data.loaders import load_splits_and_config
    config = {
        "p1": {
            "pre_op_features": {"PatientNr": "99", "age": 70.0, "bmi": 25.0},
            "ground_truth": {"reintervention": 0},
        }
    }
    splits = {"train": ["p1"], "val": [], "test": []}
    clin_dir = tmp_path / "clinical_data"
    clin_dir.mkdir()
    (tmp_path / "dataset_splits.json").write_text(json.dumps(splits))
    (clin_dir / "radar_multimodal_config_v1.json").write_text(json.dumps(config))
    _, _, feature_names = load_splits_and_config(tmp_path)
    assert "PatientNr" not in feature_names
    assert "age" in feature_names


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
    """None values and invalid strings in pre_op_features should become NaN."""
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
    assert np.isnan(result_config["p1"]["pre_op_features"]["a"])
    assert np.isnan(result_config["p1"]["pre_op_features"]["b"])


# ── compute_clin_stats ────────────────────────────────────────────────────────


def test_compute_clin_stats_mean():
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 1.0, "b": 3.0}},
        "p2": {"pre_op_features": {"a": 3.0, "b": 5.0}},
    }
    mean, _, _ = compute_clin_stats(["p1", "p2"], config, ["a", "b"])
    np.testing.assert_allclose(mean, [2.0, 4.0])


def test_compute_clin_stats_std_positive():
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 0.0}},
        "p2": {"pre_op_features": {"a": 1.0}},
    }
    _, std, _ = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert std[0] > 0  # +1e-8 floor


def test_compute_clin_stats_sentinel_excluded_from_mean():
    """Sentinel values (already converted to NaN by _clean_clinical_features)
    must not influence the mean — only real observations count."""
    from src.data.loaders import compute_clin_stats
    # p1 has sentinel → NaN; p2 and p3 have real values 2.0 and 4.0
    config = {
        "p1": {"pre_op_features": {"a": np.nan}},
        "p2": {"pre_op_features": {"a": 2.0}},
        "p3": {"pre_op_features": {"a": 4.0}},
    }
    mean, std, _ = compute_clin_stats(["p1", "p2", "p3"], config, ["a"])
    assert mean[0] == pytest.approx(3.0)   # nanmean([NaN, 2, 4]) = 3.0
    assert std[0]  >  0


def test_compute_clin_stats_all_nan_feature_gets_neutral_stats():
    """A 100%-missing feature must receive mean=0, std≈1 so it normalises to 0."""
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": np.nan}},
        "p2": {"pre_op_features": {"a": np.nan}},
    }
    mean, std, _ = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert mean[0] == pytest.approx(0.0)
    assert std[0]  == pytest.approx(1.0 + 1e-8)


def test_compute_clin_stats_constant_feature():
    """Constant feature: std = 1e-8 (just the epsilon)."""
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 5.0}},
        "p2": {"pre_op_features": {"a": 5.0}},
    }
    mean, std, _ = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert mean[0] == pytest.approx(5.0)
    assert std[0]  == pytest.approx(1e-8)


def test_compute_clin_stats_missing_key_excluded_from_mean():
    """A missing key → NaN → excluded from mean (not treated as 0)."""
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {}},          # missing "a" → NaN
        "p2": {"pre_op_features": {"a": 4.0}},  # only real observation
    }
    mean, _, _ = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert mean[0] == pytest.approx(4.0)  # nanmean([NaN, 4.0]) = 4.0


def test_compute_clin_stats_returns_median():
    """Third return value is the per-feature median (NaN-excluded)."""
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": 1.0}},
        "p2": {"pre_op_features": {"a": 2.0}},
        "p3": {"pre_op_features": {"a": 9.0}},
    }
    _, _, median = compute_clin_stats(["p1", "p2", "p3"], config, ["a"])
    assert median[0] == pytest.approx(2.0)  # nanmedian([1, 2, 9]) = 2.0


def test_compute_clin_stats_all_nan_median_falls_back_to_mean():
    """100%-missing feature: median falls back to mean (both 0.0)."""
    from src.data.loaders import compute_clin_stats
    config = {
        "p1": {"pre_op_features": {"a": np.nan}},
        "p2": {"pre_op_features": {"a": np.nan}},
    }
    mean, _, median = compute_clin_stats(["p1", "p2"], config, ["a"])
    assert median[0] == pytest.approx(mean[0])


def test_normalize_clinical_fill_value_used_for_nan():
    """NormalizeClinicald uses fill_value (z-scored median) for missing entries."""
    import torch
    from src.data.loaders import NormalizeClinicald
    mean  = np.array([0.0], dtype=np.float32)
    std   = np.array([1.0], dtype=np.float32)
    fill  = np.array([0.5], dtype=np.float32)   # z-scored median fill
    tfm   = NormalizeClinicald(keys=["clin"], mean=mean, std=std, fill_value=fill)
    data  = {"clin": torch.tensor([float("nan")])}
    out   = tfm(data)["clin"]
    assert out[0].item() == pytest.approx(0.5)


def test_normalize_clinical_no_fill_defaults_to_zero():
    """Without fill_value, NaNs are filled with 0 (mean in z-space) as before."""
    import torch
    from src.data.loaders import NormalizeClinicald
    mean = np.array([0.0], dtype=np.float32)
    std  = np.array([1.0], dtype=np.float32)
    tfm  = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    data = {"clin": torch.tensor([float("nan")])}
    out  = tfm(data)["clin"]
    assert out[0].item() == pytest.approx(0.0)


def test_normalize_clinical_fill_value_only_replaces_nan():
    """Non-NaN values are z-scored normally; only NaN entries get the fill."""
    import torch
    from src.data.loaders import NormalizeClinicald
    mean  = np.array([2.0, 0.0], dtype=np.float32)
    std   = np.array([1.0, 1.0], dtype=np.float32)
    fill  = np.array([99.0, 99.0], dtype=np.float32)
    tfm   = NormalizeClinicald(keys=["clin"], mean=mean, std=std, fill_value=fill)
    data  = {"clin": torch.tensor([4.0, float("nan")])}
    out   = tfm(data)["clin"]
    assert out[0].item() == pytest.approx(2.0)   # (4 - 2) / 1
    assert out[1].item() == pytest.approx(99.0)  # NaN → fill


# ── build_data_list ───────────────────────────────────────────────────────────


def test_build_data_list_basic(tmp_path):
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {"a": 1.0}, "ground_truth": {"reintervention": 0},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
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
                     "ground_truth": {"reintervention": 1},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
    result = build_data_list(["p1"], tmp_path, config, feat_keys)
    assert result[0]["clin"].shape == (3,)


def test_build_data_list_aaa_defaults(tmp_path):
    """Metrics file absent → all-zeros list of length 5."""
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {}, "ground_truth": {"reintervention": 0},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
    result = build_data_list(["p1"], tmp_path, config, [])
    assert result[0]["aaa_metrics"] == [0.0] * 5


def test_build_data_list_reint_label(tmp_path):
    from src.data.loaders import build_data_list
    _stub = {"scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}
    config = {
        "p1": {"pre_op_features": {}, "ground_truth": {"reintervention": 1}, **_stub},
        "p2": {"pre_op_features": {}, "ground_truth": {"reintervention": 0}, **_stub},
    }
    result = build_data_list(["p1", "p2"], tmp_path, config, [])
    assert result[0]["reintervention_label"][0] == 1.0
    assert result[1]["reintervention_label"][0] == 0.0


def test_build_data_list_with_complications(tmp_path):
    from src.data.loaders import build_data_list
    comps  = ["endoleak_1a", "migration"]
    config = {"p1": {"pre_op_features": {}, "ground_truth":
                     {"reintervention": 1, "endoleak_1a": 1},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
    result = build_data_list(["p1"], tmp_path, config, [], complications=comps)
    assert "complication_labels" in result[0]
    labels = result[0]["complication_labels"]
    assert len(labels) == 2
    assert labels[0] == 1.0   # endoleak_1a present
    assert labels[1] == -1.0  # migration not in ground_truth → -1


def test_build_data_list_without_complications_no_comp_key(tmp_path):
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {}, "ground_truth": {"reintervention": 0},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
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


def test_normalize_clinicald_nan_imputed_to_zero():
    """NaN in the raw clin vector (missing / sentinel) must become 0.0 after
    normalisation — i.e. z = 0 = the training mean, the neutral imputation."""
    from src.data.loaders import NormalizeClinicald
    mean = np.array([10.0, 2.0], dtype=np.float32)
    std  = np.array([ 5.0, 1.0], dtype=np.float32)
    t    = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    out  = t({"clin": np.array([np.nan, 4.0], dtype=np.float32)})
    assert out["clin"][0].item() == pytest.approx(0.0)   # NaN → z=0
    assert out["clin"][1].item() == pytest.approx(2.0)   # (4-2)/1 = 2


def test_normalize_clinicald_sentinel_patient_gets_z_zero():
    """End-to-end: a patient whose feature is a sentinel (9999 → NaN after
    _clean_clinical_features) must get z=0 for that feature after normalisation,
    regardless of the true feature mean and std."""
    from src.data.loaders import NormalizeClinicald
    # Suppose feature "FEV" has mean=80, std=20 on real observations.
    # A patient with sentinel 9999 should be imputed to z=0 (= mean).
    mean = np.array([80.0], dtype=np.float32)
    std  = np.array([20.0], dtype=np.float32)
    t    = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    # NaN represents the sentinel that _clean_clinical_features already converted
    out  = t({"clin": np.array([np.nan], dtype=np.float32)})
    assert out["clin"][0].item() == pytest.approx(0.0)


def test_normalize_clinicald_output_is_finite_with_nan_input():
    """No NaN must survive into the model — output must be fully finite."""
    from src.data.loaders import NormalizeClinicald
    mean = np.zeros(5, dtype=np.float32)
    std  = np.ones(5,  dtype=np.float32)
    t    = NormalizeClinicald(keys=["clin"], mean=mean, std=std)
    clin = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    out  = t({"clin": clin})
    assert torch.isfinite(out["clin"]).all()


def test_build_data_list_preserves_nan_for_missing_feature(tmp_path):
    """build_data_list must NOT call nan_to_num on clin — NaN must reach
    NormalizeClinicald so imputation happens in z-space, not raw space."""
    from src.data.loaders import build_data_list
    config = {"p1": {"pre_op_features": {"a": np.nan, "b": 3.0},
                     "ground_truth": {"reintervention": 0},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
    result = build_data_list(["p1"], tmp_path, config, ["a", "b"])
    clin = result[0]["clin"]
    assert np.isnan(clin[0]), "NaN feature must propagate to clin array"
    assert clin[1] == pytest.approx(3.0)


def test_build_data_list_sentinel_in_raw_config_reaches_array_as_nan(tmp_path):
    """A sentinel value already converted by _clean_clinical_features (→ NaN)
    must appear as NaN in the built clin array — not as 9999 or 0."""
    from src.data.loaders import build_data_list
    # Simulate post-cleaning config (as returned by load_splits_and_config)
    config = {"p1": {"pre_op_features": {"fev": np.nan, "age": 65.0},
                     "ground_truth": {"reintervention": 0},
                     "scans": {"pre_op": {"nifti": "dummy.nii.gz"}}}}
    result = build_data_list(["p1"], tmp_path, config, ["fev", "age"])
    assert np.isnan(result[0]["clin"][0])
    assert result[0]["clin"][1] == pytest.approx(65.0)


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
            "scans": {"pre_op": {"nifti": "dummy.nii.gz"}},
        }
        for i, pid in enumerate(pids_train + pids_val)
    }
    feature_names = ["feat_a", "feat_b"]
    clin_mean = np.zeros(2, dtype=np.float32)
    clin_std  = np.ones(2,  dtype=np.float32)
    return splits, config, feature_names, clin_mean, clin_std


# ── _make_weighted_sampler ────────────────────────────────────────────────────


def _make_sample_list(n_pos: int, n_neg: int) -> list:
    """Build a minimal train_files list with reintervention_label entries."""
    import numpy as np
    samples = []
    for _ in range(n_pos):
        samples.append({"reintervention_label": np.array([1.0], dtype=np.float32)})
    for _ in range(n_neg):
        samples.append({"reintervention_label": np.array([0.0], dtype=np.float32)})
    return samples


def test_make_weighted_sampler_returns_correct_type():
    from torch.utils.data import WeightedRandomSampler
    from src.data.loaders import _make_weighted_sampler
    sampler = _make_weighted_sampler(_make_sample_list(3, 7))
    assert isinstance(sampler, WeightedRandomSampler)


def test_make_weighted_sampler_count_equals_dataset_length():
    from src.data.loaders import _make_weighted_sampler
    samples = _make_sample_list(4, 6)
    sampler = _make_weighted_sampler(samples)
    assert sampler.num_samples == len(samples)


def test_make_weighted_sampler_positive_weight_equals_one_over_n_pos():
    from src.data.loaders import _make_weighted_sampler
    n_pos, n_neg = 3, 7
    samples = _make_sample_list(n_pos, n_neg)
    sampler = _make_weighted_sampler(samples)
    weights = list(sampler.weights.numpy())
    # first n_pos entries should have weight 1/n_pos
    assert all(abs(w - 1.0 / n_pos) < 1e-6 for w in weights[:n_pos])


def test_make_weighted_sampler_negative_weight_equals_one_over_n_neg():
    from src.data.loaders import _make_weighted_sampler
    n_pos, n_neg = 3, 7
    samples = _make_sample_list(n_pos, n_neg)
    sampler = _make_weighted_sampler(samples)
    weights = list(sampler.weights.numpy())
    assert all(abs(w - 1.0 / n_neg) < 1e-6 for w in weights[n_pos:])


# ── get_cached_transforms — transform pipeline contents ───────────────────────


def test_cached_transforms_uses_spatial_pad_and_crop():
    """get_cached_transforms must use SpatialPadd + CenterSpatialCropd, not Resized."""
    from monai.transforms import CenterSpatialCropd, SpatialPadd
    from src.data.loaders import get_cached_transforms

    tfm = get_cached_transforms()
    transform_types = [type(t) for t in tfm.transforms]
    assert SpatialPadd in transform_types
    assert CenterSpatialCropd in transform_types


def test_cached_transforms_has_no_resized():
    """Resized must NOT appear in the cached transform pipeline (replaced by pad+crop)."""
    from monai.transforms import Resized
    from src.data.loaders import get_cached_transforms

    tfm = get_cached_transforms()
    transform_types = [type(t) for t in tfm.transforms]
    assert Resized not in transform_types


# ── build_loaders ─────────────────────────────────────────────────────────────


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_returns_two_dataloaders(mock_cache, mock_dl, tmp_path):
    """build_loaders must return a (train_loader, val_loader) tuple."""
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 2)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    result = build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, batch_size=2,
    )
    assert isinstance(result, tuple) and len(result) == 2


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_train_uses_sampler(mock_cache, mock_dl, tmp_path):
    """Train DataLoader must use WeightedRandomSampler (not shuffle) for balanced batches."""
    from torch.utils.data import WeightedRandomSampler
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 4)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, batch_size=2,
    )
    train_kwargs = mock_dl.call_args_list[0].kwargs
    assert isinstance(train_kwargs.get("sampler"), WeightedRandomSampler)
    assert "shuffle" not in train_kwargs


@patch("torch.utils.data.DataLoader")
@patch("src.data.loaders.CacheDataset")
def test_build_loaders_val_uses_shuffle_false(mock_cache, mock_dl, tmp_path):
    """Val DataLoader must be built with shuffle=False (deterministic evaluation order)."""
    from src.data.loaders import build_loaders
    mock_cache.return_value = MagicMock(__len__=lambda s: 2)
    splits, config, feature_names, clin_mean, clin_std = _build_loaders_args(tmp_path)

    build_loaders(
        splits, config, feature_names, clin_mean, clin_std,
        base_dir=tmp_path, batch_size=2,
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
        base_dir=tmp_path, batch_size=4,
    )
    assert mock_dl.call_args_list[0].kwargs["batch_size"] == 4
    assert mock_dl.call_args_list[1].kwargs["batch_size"] == 4


# ── augmentation pipeline tests ───────────────────────────────────────────────
#
# These tests verify the training augmentation pipeline defined in
# get_train_runtime_transforms is correctly assembled and behaves as expected:
#   1. Pipeline composition  — all augment classes present, correct count
#   2. Key coverage          — spatial ops touch ct+lbl+dst; intensity ops touch ct only
#   3. Ordering              — spatial augments precede BuildCTAd
#   4. Flip correctness      — each axis flips the right spatial dimension
#   5. Intensity isolation   — intensity transforms do not alter lbl or dst
#   6. Shape preservation    — zoom and affine keep_size returns original shape
#   7. Stochasticity         — two forward passes on the same volume differ

from monai.transforms import (
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
)

_SPATIAL_TYPES   = (RandFlipd, RandRotate90d, RandAffined, RandZoomd)
_INTENSITY_TYPES = (
    RandAdjustContrastd, RandShiftIntensityd, RandScaleIntensityd,
    RandBiasFieldd, RandGaussianNoised, RandGaussianSmoothd,
)


def _aug_vol(size=(8, 8, 8)):
    """Minimal {ct, lbl, dst} dict for augmentation tests (no NIfTI I/O needed)."""
    return {
        "ct":  torch.rand(1, *size),
        "lbl": torch.randint(0, 3, (1, *size)).float(),
        "dst": torch.rand(1, *size),
    }


def _train_aug_tfm():
    from src.data.loaders import get_train_runtime_transforms
    return get_train_runtime_transforms(
        np.zeros(CLINICAL_DIM, dtype=np.float32),
        np.ones(CLINICAL_DIM, dtype=np.float32),
    )


def _val_aug_tfm():
    from src.data.loaders import get_val_runtime_transforms
    return get_val_runtime_transforms(
        np.zeros(CLINICAL_DIM, dtype=np.float32),
        np.ones(CLINICAL_DIM, dtype=np.float32),
    )


# ── 1. Pipeline composition ───────────────────────────────────────────────────

def test_train_pipeline_has_all_spatial_transform_types():
    """All four spatial augment classes must appear in the training Compose."""
    types = {type(t) for t in _train_aug_tfm().transforms}
    for cls in _SPATIAL_TYPES:
        assert cls in types, f"{cls.__name__} missing from training pipeline"


def test_train_pipeline_has_all_intensity_transform_types():
    """All six intensity augment classes must appear in the training Compose."""
    types = {type(t) for t in _train_aug_tfm().transforms}
    for cls in _INTENSITY_TYPES:
        assert cls in types, f"{cls.__name__} missing from training pipeline"


def test_train_pipeline_has_exactly_three_flip_transforms():
    """Exactly three RandFlipd instances must be present (axes 0, 1, 2)."""
    flips = [t for t in _train_aug_tfm().transforms if isinstance(t, RandFlipd)]
    assert len(flips) == 3


def test_val_pipeline_has_no_rand_augmentations():
    """Val Compose must contain no Rand* augmentation transforms."""
    rand_types = _SPATIAL_TYPES + _INTENSITY_TYPES
    for t in _val_aug_tfm().transforms:
        assert not isinstance(t, rand_types), (
            f"Val pipeline must not contain {type(t).__name__}"
        )


# ── 2. Key coverage ───────────────────────────────────────────────────────────

def test_spatial_transforms_include_lbl_and_dst():
    """Every spatial augment must operate on lbl and dst (not ct alone)."""
    for t in _train_aug_tfm().transforms:
        if isinstance(t, _SPATIAL_TYPES):
            assert "lbl" in t.keys, f"{type(t).__name__} missing 'lbl' key"
            assert "dst" in t.keys, f"{type(t).__name__} missing 'dst' key"


def test_intensity_transforms_are_ct_only():
    """Every intensity augment must operate on ct only — never on lbl or dst."""
    for t in _train_aug_tfm().transforms:
        if isinstance(t, _INTENSITY_TYPES):
            assert list(t.keys) == ["ct"], (
                f"{type(t).__name__} keys {list(t.keys)} must be ['ct'] only"
            )


# ── 3. Ordering ───────────────────────────────────────────────────────────────

def test_spatial_augments_precede_build_ctad():
    """All spatial augments must appear before BuildCTAd in the pipeline."""
    from src.data.loaders import BuildCTAd
    transforms = _train_aug_tfm().transforms
    build_idx = next(i for i, t in enumerate(transforms) if isinstance(t, BuildCTAd))
    for i, t in enumerate(transforms):
        if isinstance(t, _SPATIAL_TYPES):
            assert i < build_idx, (
                f"{type(t).__name__} at index {i} appears after BuildCTAd at {build_idx}"
            )


def test_intensity_augments_precede_build_ctad():
    """All intensity augments must also appear before BuildCTAd."""
    from src.data.loaders import BuildCTAd
    transforms = _train_aug_tfm().transforms
    build_idx = next(i for i, t in enumerate(transforms) if isinstance(t, BuildCTAd))
    for i, t in enumerate(transforms):
        if isinstance(t, _INTENSITY_TYPES):
            assert i < build_idx, (
                f"{type(t).__name__} at index {i} appears after BuildCTAd at {build_idx}"
            )


# ── 4. Flip correctness ───────────────────────────────────────────────────────

def _asymmetric_vol(size=(4, 6, 8)):
    """Volume whose spatial dims are all different to make axis errors obvious."""
    ct  = torch.arange(1 * size[0] * size[1] * size[2]).float().reshape(1, *size)
    lbl = ct.clone()
    dst = ct.clone()
    return {"ct": ct, "lbl": lbl, "dst": dst}


def test_flip_axis0_reverses_first_spatial_dim():
    """RandFlipd(spatial_axis=0, prob=1.0) must reverse tensor dimension 1."""
    data = _asymmetric_vol()
    t = RandFlipd(keys=["ct", "lbl", "dst"], spatial_axis=0, prob=1.0)
    out = t(data)
    expected = data["ct"].flip(1)
    assert torch.allclose(out["ct"], expected)


def test_flip_axis1_reverses_second_spatial_dim():
    """RandFlipd(spatial_axis=1, prob=1.0) must reverse tensor dimension 2."""
    data = _asymmetric_vol()
    t = RandFlipd(keys=["ct", "lbl", "dst"], spatial_axis=1, prob=1.0)
    out = t(data)
    expected = data["ct"].flip(2)
    assert torch.allclose(out["ct"], expected)


def test_flip_axis2_reverses_third_spatial_dim():
    """RandFlipd(spatial_axis=2, prob=1.0) must reverse tensor dimension 3."""
    data = _asymmetric_vol()
    t = RandFlipd(keys=["ct", "lbl", "dst"], spatial_axis=2, prob=1.0)
    out = t(data)
    expected = data["ct"].flip(3)
    assert torch.allclose(out["ct"], expected)


def test_flip_applied_consistently_to_ct_and_lbl():
    """When ct and lbl start identical, a flip must leave them identical afterwards."""
    ct  = torch.rand(1, 6, 6, 6)
    data = {"ct": ct.clone(), "lbl": ct.clone(), "dst": torch.zeros(1, 6, 6, 6)}
    t = RandFlipd(keys=["ct", "lbl", "dst"], spatial_axis=0, prob=1.0)
    out = t(data)
    assert torch.allclose(out["ct"], out["lbl"])


def test_flip_lbl_values_are_preserved_after_flip():
    """Flip must permute voxels, not alter label values (no interpolation on lbl)."""
    lbl = torch.randint(0, 3, (1, 8, 8, 8)).float()
    data = {"ct": torch.rand(1, 8, 8, 8), "lbl": lbl.clone(), "dst": torch.rand(1, 8, 8, 8)}
    t = RandFlipd(keys=["ct", "lbl", "dst"], spatial_axis=0, prob=1.0)
    out = t(data)
    # Flatten + convert MetaTensor → plain tensor before sorting to avoid type mismatch
    out_sorted  = torch.as_tensor(out["lbl"]).float().flatten().sort().values
    orig_sorted = lbl.flatten().sort().values
    assert torch.allclose(out_sorted, orig_sorted), "Flip changed label values (not just order)"


# ── 5. Intensity isolation ────────────────────────────────────────────────────

def _intensity_iso(cls, **kwargs):
    """Apply a ct-only intensity augment (prob=1.0) and check lbl+dst are unchanged."""
    data     = _aug_vol(size=(8, 8, 8))
    lbl_orig = data["lbl"].clone()
    dst_orig = data["dst"].clone()
    t = cls(keys=["ct"], prob=1.0, **kwargs)
    out = t(data)
    assert torch.allclose(out["lbl"], lbl_orig), f"{cls.__name__} modified lbl"
    assert torch.allclose(out["dst"], dst_orig), f"{cls.__name__} modified dst"


def test_gaussian_noise_does_not_modify_lbl_or_dst():
    _intensity_iso(RandGaussianNoised, mean=0.0, std=0.5)


def test_scale_intensity_does_not_modify_lbl_or_dst():
    _intensity_iso(RandScaleIntensityd, factors=0.5)


def test_shift_intensity_does_not_modify_lbl_or_dst():
    _intensity_iso(RandShiftIntensityd, offsets=0.3)


def test_adjust_contrast_does_not_modify_lbl_or_dst():
    _intensity_iso(RandAdjustContrastd, gamma=(2.0, 2.0))


def test_bias_field_does_not_modify_lbl_or_dst():
    _intensity_iso(RandBiasFieldd, coeff_range=(0.1, 0.3))


def test_gaussian_smooth_does_not_modify_lbl_or_dst():
    _intensity_iso(
        RandGaussianSmoothd,
        sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.25, 0.5),
    )


# ── 6. Shape preservation ─────────────────────────────────────────────────────

def test_zoom_keep_size_preserves_spatial_shape():
    """RandZoomd with keep_size=True must output the same spatial shape as input."""
    size = (16, 16, 16)
    data = _aug_vol(size=size)
    t = RandZoomd(
        keys=["ct", "lbl", "dst"],
        min_zoom=0.85, max_zoom=1.15, prob=1.0,
        mode=("trilinear", "nearest", "trilinear"),
        keep_size=True, padding_mode="edge",
    )
    out = t(data)
    for key in ("ct", "lbl", "dst"):
        assert out[key].shape[1:] == size, (
            f"zoom changed {key} shape: got {out[key].shape[1:]}"
        )


def test_affine_preserves_spatial_shape():
    """RandAffined must return the same spatial shape as its input."""
    size = (16, 16, 16)
    data = _aug_vol(size=size)
    t = RandAffined(
        keys=["ct", "lbl", "dst"], prob=1.0,
        rotate_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest", "bilinear"),
        padding_mode="border",
    )
    out = t(data)
    for key in ("ct", "lbl", "dst"):
        assert out[key].shape[1:] == size, (
            f"affine changed {key} shape: got {out[key].shape[1:]}"
        )


def test_zoom_lbl_stays_in_valid_range_after_zoom():
    """After zoom with nearest-neighbour interpolation, lbl values must stay in {0,1,2}."""
    lbl = torch.randint(0, 3, (1, 16, 16, 16)).float()
    data = {"ct": torch.rand(1, 16, 16, 16), "lbl": lbl, "dst": torch.rand(1, 16, 16, 16)}
    t = RandZoomd(
        keys=["ct", "lbl", "dst"],
        min_zoom=0.9, max_zoom=1.1, prob=1.0,
        mode=("trilinear", "nearest", "trilinear"),
        keep_size=True, padding_mode="edge",
    )
    out = t(data)
    unique_vals = out["lbl"].unique()
    assert all(v in {0.0, 1.0, 2.0} for v in unique_vals.tolist()), (
        f"zoom produced unexpected lbl values: {unique_vals.tolist()}"
    )


# ── 7. Stochasticity ──────────────────────────────────────────────────────────

def test_augmentation_pipeline_produces_varied_ct():
    """Two forward passes through the training pipeline on the same ct must differ."""
    data1 = _aug_vol(size=(8, 8, 8))
    # Deep-copy so both calls start from identical input
    data2 = {k: v.clone() for k, v in data1.items()}
    # Add required non-image keys so the full Compose can complete
    clin = np.zeros(CLINICAL_DIM, dtype=np.float32)
    for d in (data1, data2):
        d["clin"]                 = clin.copy()
        d["aaa_metrics"]          = np.zeros(5, dtype=np.float32)
        d["reintervention_label"] = np.array([0.0], dtype=np.float32)

    tfm = _train_aug_tfm()
    out1 = tfm(data1)
    out2 = tfm(data2)
    # With many augments at various probabilities, identical output is astronomically unlikely
    assert not torch.allclose(out1["cta"], out2["cta"]), (
        "Two augmented passes produced identical CTA — augmentation may not be firing"
    )
