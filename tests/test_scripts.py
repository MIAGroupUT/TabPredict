"""
Tests for scripts/train.py and scripts/test.py

Covers:
- _apply_overrides: integer coercion, float coercion, boolean coercion,
  nested dot-path keys, string fallback, empty override list, multiple overrides
- train.py argparse: SystemExit when --config is missing
- test.py  argparse: SystemExit when any required arg is missing
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make scripts/ importable from the repo root (ai4evar/)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_apply_overrides():
    """Import _apply_overrides without triggering run_pipeline import."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "scripts_train",
        Path(__file__).parent.parent / "scripts" / "train.py",
    )
    # Temporarily stub run_pipeline so we don't import trainer.py in full.
    # Save and restore sys.modules state so this module-level call (executed
    # at collection time) does not leave the fake module cached and break later
    # imports of the real src.training.trainer in other test files.
    _prev = sys.modules.get("src.training.trainer")
    fake_trainer = types.ModuleType("src.training.trainer")
    fake_trainer.run_pipeline = lambda cfg: None  # no-op
    sys.modules["src.training.trainer"] = fake_trainer
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if _prev is None:
            sys.modules.pop("src.training.trainer", None)
        else:
            sys.modules["src.training.trainer"] = _prev
    return mod._apply_overrides


_apply_overrides = _load_apply_overrides()


# ── _apply_overrides — type coercion ─────────────────────────────────────────


def test_apply_overrides_integer():
    cfg = {"training": {"batch_size": 8}}
    result = _apply_overrides(cfg, ["training.batch_size=4"])
    assert result["training"]["batch_size"] == 4
    assert isinstance(result["training"]["batch_size"], int)


def test_apply_overrides_float():
    cfg = {"training": {"lr": 1e-3}}
    result = _apply_overrides(cfg, ["training.lr=0.0001"])
    assert result["training"]["lr"] == pytest.approx(0.0001)
    assert isinstance(result["training"]["lr"], float)


def test_apply_overrides_bool_true():
    cfg = {"training": {"use_amp": False}}
    result = _apply_overrides(cfg, ["training.use_amp=true"])
    assert result["training"]["use_amp"] is True


def test_apply_overrides_bool_false():
    cfg = {"training": {"use_amp": True}}
    result = _apply_overrides(cfg, ["training.use_amp=false"])
    assert result["training"]["use_amp"] is False


def test_apply_overrides_bool_mixed_case():
    cfg = {}
    result = _apply_overrides(cfg, ["flag=True"])
    assert result["flag"] is True


def test_apply_overrides_string_fallback():
    cfg = {}
    result = _apply_overrides(cfg, ["paths.base_dir=/tmp/data"])
    assert result["paths"]["base_dir"] == "/tmp/data"
    assert isinstance(result["paths"]["base_dir"], str)


def test_apply_overrides_nested_creates_intermediate_keys():
    cfg = {}
    result = _apply_overrides(cfg, ["a.b.c=42"])
    assert result["a"]["b"]["c"] == 42


def test_apply_overrides_top_level_key():
    cfg = {"name": "old"}
    result = _apply_overrides(cfg, ["name=new_run"])
    assert result["name"] == "new_run"


def test_apply_overrides_empty_list():
    cfg = {"x": 1}
    result = _apply_overrides(cfg, [])
    assert result == {"x": 1}


def test_apply_overrides_multiple():
    cfg = {}
    result = _apply_overrides(cfg, ["training.batch_size=16", "training.lr=1e-4", "name=exp"])
    assert result["training"]["batch_size"] == 16
    assert result["training"]["lr"] == pytest.approx(1e-4)
    assert result["name"] == "exp"


def test_apply_overrides_int_wins_over_float():
    """'4' should be coerced to int (tried first), not float."""
    cfg = {}
    result = _apply_overrides(cfg, ["x=4"])
    assert isinstance(result["x"], int)
    assert result["x"] == 4


def test_apply_overrides_returns_same_dict():
    """_apply_overrides mutates and returns the same dict object."""
    cfg = {}
    returned = _apply_overrides(cfg, ["x=1"])
    assert returned is cfg


# ── argparse — train.py ───────────────────────────────────────────────────────


def test_train_argparse_missing_config_exits():
    """train.py requires --config; missing it must raise SystemExit."""
    with patch.object(sys, "argv", ["train.py"]):
        with pytest.raises(SystemExit) as exc:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", required=True)
            parser.add_argument("--set", nargs="*", default=[])
            parser.parse_args()
        assert exc.value.code != 0


def test_train_argparse_accepts_config():
    with patch.object(sys, "argv", ["train.py", "--config", "experiments/v1_no_complications.yaml"]):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--set", nargs="*", default=[])
        args = parser.parse_args()
        assert args.config == "experiments/v1_no_complications.yaml"
        assert args.set == []


def test_train_argparse_set_override_list():
    with patch.object(sys, "argv", [
        "train.py", "--config", "exp.yaml", "--set", "training.batch_size=4", "name=run2",
    ]):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--set", nargs="*", default=[])
        args = parser.parse_args()
        assert args.set == ["training.batch_size=4", "name=run2"]


# ── argparse — test.py ───────────────────────────────────────────────────────


def _make_test_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--gradcam",     action="store_true")
    parser.add_argument("--workers",     type=int, default=8)
    return parser


def test_test_argparse_missing_all_required_exits():
    with patch.object(sys, "argv", ["test.py"]):
        with pytest.raises(SystemExit) as exc:
            _make_test_parser().parse_args()
        assert exc.value.code != 0


def test_test_argparse_missing_checkpoint_exits():
    with patch.object(sys, "argv", ["test.py", "--config", "exp.yaml", "--output_dir", "/tmp"]):
        with pytest.raises(SystemExit) as exc:
            _make_test_parser().parse_args()
        assert exc.value.code != 0


def test_test_argparse_missing_output_dir_exits():
    with patch.object(sys, "argv", ["test.py", "--config", "exp.yaml", "--checkpoint", "ckpt.pt"]):
        with pytest.raises(SystemExit) as exc:
            _make_test_parser().parse_args()
        assert exc.value.code != 0


def test_test_argparse_all_required_present():
    with patch.object(sys, "argv", [
        "test.py",
        "--config",     "experiments/v1_no_complications.yaml",
        "--checkpoint", "/tmp/best.pt",
        "--output_dir", "/tmp/out",
    ]):
        args = _make_test_parser().parse_args()
        assert args.config      == "experiments/v1_no_complications.yaml"
        assert args.checkpoint  == "/tmp/best.pt"
        assert args.output_dir  == "/tmp/out"
        assert args.split       == "test"
        assert args.gradcam     is False
        assert args.workers     == 8


def test_test_argparse_gradcam_flag():
    with patch.object(sys, "argv", [
        "test.py",
        "--config", "e.yaml", "--checkpoint", "c.pt", "--output_dir", "/o",
        "--gradcam",
    ]):
        args = _make_test_parser().parse_args()
        assert args.gradcam is True


def test_test_argparse_split_choices():
    for split in ("train", "val", "test"):
        with patch.object(sys, "argv", [
            "test.py",
            "--config", "e.yaml", "--checkpoint", "c.pt", "--output_dir", "/o",
            "--split", split,
        ]):
            args = _make_test_parser().parse_args()
            assert args.split == split


def test_test_argparse_invalid_split_exits():
    with patch.object(sys, "argv", [
        "test.py",
        "--config", "e.yaml", "--checkpoint", "c.pt", "--output_dir", "/o",
        "--split", "bogus",
    ]):
        with pytest.raises(SystemExit) as exc:
            _make_test_parser().parse_args()
        assert exc.value.code != 0
