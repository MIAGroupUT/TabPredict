#!/usr/bin/env python3
"""
Test / inference entry point.

Usage
─────
python scripts/test.py \
    --config     experiments/v1_no_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_no_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_no_complications \
    [--split test] [--gradcam] [--workers 8]

python scripts/test.py \
    --config     experiments/v1_with_complications.yaml \
    --checkpoint /home/ranema/AI4EVAR/checkpoints/v1_with_complications/best_stage_b.pt \
    --output_dir /home/ranema/AI4EVAR/test_results/v1_with_complications \
    --gradcam
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.runner import run_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AAA reintervention risk model on test set")
    parser.add_argument(
        "--config", required=True,
        help="Experiment YAML used for training (e.g. experiments/v1_no_complications.yaml)",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained .pt checkpoint (e.g. checkpoints/v1_no_complications/best_stage_b.pt)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root directory to write per-patient results and overall metrics",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Compute and save GradCAM PNGs per patient (~10–30 s each, needs backward pass)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="DataLoader worker processes (default: 8)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Evaluating : {cfg['name']}  (version {cfg.get('version', '?')})")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"GradCAM    : {args.gradcam}\n")

    run_test(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        gradcam=args.gradcam,
        split=args.split,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
