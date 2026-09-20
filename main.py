#!/usr/bin/env python3
"""
Main entry point for the Skin Cancer Detection System.

Usage:
    python main.py train
    python main.py test
    python main.py predict --image sample.jpg
    python main.py app
    python main.py download
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explainable Hybrid Deep Learning Framework for Early Skin Cancer Detection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--segmentation-only", action="store_true")
    train_parser.add_argument("--classification-only", action="store_true")
    train_parser.add_argument("--cross-validation", action="store_true")
    train_parser.add_argument("--optuna", action="store_true")
    train_parser.add_argument("--download", action="store_true")
    train_parser.add_argument("--export", action="store_true")

    test_parser = subparsers.add_parser("test", help="Evaluate model")
    test_parser.add_argument("--config", default="configs/default.yaml")
    test_parser.add_argument("--checkpoint", default=None)

    predict_parser = subparsers.add_parser("predict", help="Predict single image")
    predict_parser.add_argument("--config", default="configs/default.yaml")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--age", type=float, default=50.0)
    predict_parser.add_argument("--gender", default="unknown")
    predict_parser.add_argument("--location", default="unknown")

    subparsers.add_parser("app", help="Launch Streamlit web application")
    subparsers.add_parser("download", help="Download dataset")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", args.config]
        if args.segmentation_only:
            cmd.append("--segmentation-only")
        if args.classification_only:
            cmd.append("--classification-only")
        if args.cross_validation:
            cmd.append("--cross-validation")
        if args.optuna:
            cmd.append("--optuna")
        if args.download:
            cmd.append("--download")
        if args.export:
            cmd.append("--export")
        subprocess.run(cmd, check=True)

    elif args.command == "test":
        cmd = [sys.executable, str(PROJECT_ROOT / "test.py"), "--config", args.config]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        subprocess.run(cmd, check=True)

    elif args.command == "predict":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--config",
            args.config,
            "--image",
            args.image,
            "--age",
            str(args.age),
            "--gender",
            args.gender,
            "--location",
            args.location,
        ]
        subprocess.run(cmd, check=True)

    elif args.command == "app":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py")],
            check=True,
        )

    elif args.command == "download":
        sys.path.insert(0, str(PROJECT_ROOT))
        from utils.config import load_config
        from utils.download import DatasetDownloader

        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        downloader = DatasetDownloader(config)
        path = downloader.download()
        print(f"Dataset ready at: {path}")


if __name__ == "__main__":
    main()
