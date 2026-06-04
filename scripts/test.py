#!/usr/bin/env python3
"""
Test / inference entry point.

Usage
─────
# Evaluate a single checkpoint on the test split (default, GradCAM on):
python scripts/test.py \
    --config     experiments/v1_no_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications
# Results land in:  output_dir/test/

# Evaluate on a specific split:
python scripts/test.py ... --split val
# Results land in:  output_dir/val/

# Evaluate on ALL splits × ALL stage checkpoints in one go:
python scripts/test.py \
    --config     experiments/v1_no_complications.yaml \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
    --split all
# Results land in:
#   output_dir/train/stage_a/   output_dir/train/stage_b/
#   output_dir/val/stage_a/     output_dir/val/stage_b/
#   output_dir/test/stage_a/    output_dir/test/stage_b/

# Override a config value (e.g. point to a different checkpoint dir):
python scripts/test.py \
    --config     experiments/v1_no_complications.yaml \
    --set paths.checkpoint_dir=/home/ranema/AI4EVAR/checkpoints/my_run \
    --output_dir /home/ranema/AI4EVAR/test_results/my_run \
    --split all

# Skip GradCAM for a fast metrics-only run:
python scripts/test.py ... --no-gradcam
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.runner import run_test

ALL_SPLITS = ["train", "val", "test"]


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-path overrides (key=value) onto a nested config dict."""
    for override in overrides:
        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        for coerce in (int, float):
            try:
                value = coerce(raw_value); break
            except ValueError:
                pass
        else:
            value = {"true": True, "false": False}.get(raw_value.lower(), raw_value)
        node[keys[-1]] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AAA reintervention risk model on one or all data splits"
    )
    parser.add_argument(
        "--config", required=True,
        help="Experiment YAML used for training (e.g. experiments/v1_no_complications.yaml)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Path to a specific .pt checkpoint. "
            "If omitted, all best_stage_*.pt files in cfg.paths.checkpoint_dir "
            "are evaluated automatically, each into output_dir/stage_<x>/."
        ),
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root directory for results. Split and stage are appended automatically "
             "so results from different runs never overwrite each other.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test", "all"],
        help="Which dataset split to evaluate. 'all' runs train+val+test (default: test)",
    )
    parser.add_argument(
        "--no-gradcam", dest="gradcam", action="store_false",
        help="Disable GradCAM PNGs (enabled by default)",
    )
    parser.set_defaults(gradcam=True)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="DataLoader worker processes (default: 8)",
    )
    parser.add_argument(
        "--set", nargs="*", default=[],
        metavar="key=value",
        help="Override any config value, e.g. --set paths.checkpoint_dir=/my/ckpts",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.set:
        cfg = _apply_overrides(cfg, args.set)

    output_dir = Path(args.output_dir)
    splits = ALL_SPLITS if args.split == "all" else [args.split]

    if args.checkpoint:
        # ── single checkpoint mode ─────────────────────────────────────────────
        for split in splits:
            _run_one(cfg, args.checkpoint, output_dir / split, split, args)
    else:
        # ── auto-discover all stage checkpoints ───────────────────────────────
        ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        checkpoints = sorted(ckpt_dir.glob("best_stage_*.pt"))
        if not checkpoints:
            print(f"No best_stage_*.pt files found in {ckpt_dir}")
            sys.exit(1)

        print(f"Found {len(checkpoints)} checkpoint(s) in {ckpt_dir}:")
        for c in checkpoints:
            print(f"  {c.name}")
        print()

        for ckpt_path in checkpoints:
            # e.g. best_stage_b.pt  →  <split>/stage_b
            stage_tag = ckpt_path.stem.replace("best_", "")   # "stage_a" / "stage_b"
            for split in splits:
                stage_out = output_dir / split / stage_tag
                print(f"\n{'='*62}")
                print(f"EVALUATING  {ckpt_path.name}  [{split}]  →  {stage_out}")
                print(f"{'='*62}")
                _run_one(cfg, str(ckpt_path), stage_out, split, args)


def _run_one(cfg: dict, checkpoint: str, output_dir: Path, split: str, args) -> None:
    print(f"Config     : {cfg['name']}  (version {cfg.get('version', '?')})")
    print(f"Checkpoint : {checkpoint}")
    print(f"Split      : {split}")
    print(f"GradCAM    : {args.gradcam}\n")

    run_test(
        cfg=cfg,
        checkpoint_path=checkpoint,
        output_dir=str(output_dir),
        gradcam=args.gradcam,
        split=split,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
