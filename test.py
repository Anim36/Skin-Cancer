#!/usr/bin/env python3
"""
Testing and evaluation script for the Skin Cancer Detection System.

Usage:
    python test.py --config configs/default.yaml
    python test.py --checkpoint checkpoints/classifier_best.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from models.hybrid_model import HybridSkinCancerModel
from utils.config import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Skin Cancer Detection model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override classifier checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    set_seed(config.get("project.seed", 42))

    if args.checkpoint:
        config.set("classification.checkpoint_name", Path(args.checkpoint).name)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / args.checkpoint
        tf.keras.models.load_model(checkpoint_path)  # validate checkpoint exists

    hybrid = HybridSkinCancerModel(config)
    train_samples, val_samples, test_samples = hybrid.load_data()

    split_map = {"train": train_samples, "val": val_samples, "test": test_samples}
    samples = split_map[args.split]

    print(f"Evaluating on {args.split} split ({len(samples)} samples)...")
    metrics = hybrid.evaluate(samples)

    print("\n=== Final Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
