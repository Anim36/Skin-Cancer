#!/usr/bin/env python3
"""
Single-image prediction script with Grad-CAM explainability.

Usage:
    python predict.py --image path/to/lesion.jpg --age 45 --gender male --location back
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.hybrid_model import HybridSkinCancerModel
from utils.config import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict skin cancer from dermoscopic image")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to skin lesion image")
    parser.add_argument("--age", type=float, default=50.0, help="Patient age")
    parser.add_argument("--gender", type=str, default="unknown", choices=["male", "female", "unknown"])
    parser.add_argument("--location", type=str, default="unknown", help="Lesion anatomical location")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    set_seed(config.get("project.seed", 42))

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hybrid = HybridSkinCancerModel(config)
    if args.checkpoint:
        import tensorflow as tf
        hybrid.classification_model = tf.keras.models.load_model(args.checkpoint, compile=False)

    result = hybrid.predict(
        image=image,
        age=args.age,
        gender=args.gender,
        location=args.location,
    )

    print("\n" + "=" * 50)
    print("SKIN CANCER DETECTION RESULT")
    print("=" * 50)
    print(f"Predicted Disease : {result['disease'].upper()}")
    print(f"Confidence Score  : {result['confidence']:.2%}")
    print(f"Risk Level        : {result['risk_level']}")
    print("\nClass Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cls:8s}: {prob:.2%}")
    if result.get("gradcam_path"):
        print(f"\nGrad-CAM saved to: {result['gradcam_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
