#!/usr/bin/env python3
"""
Training entry point.

Usage
─────
python scripts/train.py --config experiments/v1_no_complications.yaml
python scripts/train.py --config experiments/v1_with_complications.yaml

# Override any leaf value without editing the YAML:
python scripts/train.py --config experiments/v1_no_complications.yaml \
    --set training.batch_size=4 paths.checkpoint_dir=/tmp/ckpt
"""

import argparse
import sys
from pathlib import Path

import yaml

# Make the repo root importable regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import run_pipeline


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-path overrides (key=value) onto a nested config dict."""
    for override in overrides:
        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Try to coerce to int / float / bool; fall back to string
        value: object
        for coerce in (int, float):
            try:
                value = coerce(raw_value)
                break
            except ValueError:
                pass
        else:
            value = {"true": True, "false": False}.get(raw_value.lower(), raw_value)
        node[keys[-1]] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AAA reintervention risk model")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML (e.g. experiments/v1_no_complications.yaml)",
    )
    parser.add_argument(
        "--set", nargs="*", default=[],
        metavar="key=value",
        help="Override config values, e.g. --set training.batch_size=4",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.set:
        cfg = _apply_overrides(cfg, args.set)

    print(f"Starting: {cfg['name']}  (version {cfg.get('version', '?')})")
    print(f"Description: {cfg.get('description', '').strip()}\n")
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
