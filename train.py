#!/usr/bin/env python3
"""
Training script for the Skin Cancer Detection System.

Usage:
    python train.py --config configs/default.yaml
    python train.py --segmentation-only
    python train.py --classification-only
    python train.py --cross-validation
    python train.py --optuna
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, set_seed
from utils.cross_validation import CrossValidator
from utils.hyperparameter_tuning import HyperparameterTuner, suggest_hyperparameters
from models.hybrid_model import HybridSkinCancerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Skin Cancer Detection hybrid deep learning framework"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--segmentation-only", action="store_true", help="Train only U-Net segmentation")
    parser.add_argument("--classification-only", action="store_true", help="Train only classifier")
    parser.add_argument("--cross-validation", action="store_true", help="Run K-Fold cross validation")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--download", action="store_true", help="Auto-download dataset")
    parser.add_argument("--no-fine-tune", action="store_true", help="Skip fine-tuning phase")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    return parser.parse_args()


def train_fold(hybrid: HybridSkinCancerModel, train_samples, val_samples, fold_idx):
    """Training function for cross-validation folds."""
    model, history = hybrid.train_classifier(train_samples, val_samples, fine_tune=False)
    metrics = hybrid.evaluate(val_samples)
    return model, history, metrics


def optuna_objective(trial, config):
    """Optuna objective function."""
    import optuna

    params = suggest_hyperparameters(trial, config)
    for key, value in params.items():
        if key == "batch_size":
            config.set("classification.batch_size", value)
        elif key == "dropout":
            config.set("classification.dropout", value)
        elif key == "attention_type":
            config.set("classification.attention_type", value)
        elif key == "optimizer":
            config.set("training.optimizer", value)
        elif key == "loss":
            config.set("training.loss", value)
        elif key == "learning_rate":
            config.set("classification.learning_rate", value)
        elif key == "weight_decay":
            config.set("training.weight_decay", value)

    hybrid = HybridSkinCancerModel(config)
    train_samples, val_samples, _ = hybrid.load_data()
    _, _ = hybrid.train_classifier(train_samples, val_samples, fine_tune=False)
    metrics = hybrid.evaluate(val_samples)
    return metrics.get("f1", 0.0)


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    if args.download:
        config.set("download.auto_download", True)

    set_seed(config.get("project.seed", 42))
    hybrid = HybridSkinCancerModel(config)

    print("=" * 60)
    print(config.get("project.name"))
    print("=" * 60)

    if args.optuna:
        config.set("hyperparameter_tuning.enabled", True)
        tuner = HyperparameterTuner(
            config,
            objective_fn=lambda trial: optuna_objective(trial, config),
            n_trials=config.get("hyperparameter_tuning.n_trials", 20),
        )
        study = tuner.run()
        print(f"Best trial F1: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        return

    if args.cross_validation:
        config.set("cross_validation.enabled", True)
        cv = CrossValidator(config, n_folds=config.get("cross_validation.n_folds", 5))
        hybrid.dataset_manager.load()
        cv.run(lambda tr, va, fold: train_fold(hybrid, tr, va, fold))
        return

    train_samples, val_samples, test_samples = hybrid.load_data()

    if not args.classification_only:
        print("\n[1/2] Training U-Net Segmentation Model...")
        seg_model, seg_history = hybrid.train_segmentation(train_samples, val_samples)
        if seg_history:
            hybrid.visualizer.plot_training_history(seg_history, prefix="segmentation")

    if not args.segmentation_only:
        print("\n[2/2] Training Hybrid Classification Model...")
        cls_model, cls_history = hybrid.train_classifier(
            train_samples,
            val_samples,
            fine_tune=not args.no_fine_tune,
        )

        print("\n[Evaluation] Testing on held-out test set...")
        metrics = hybrid.evaluate(test_samples)

        print("\n[Comparison] Loss and Optimizer options:")
        comparison = hybrid.compare_losses_and_optimizers()
        print(comparison)

        if args.export or config.get("export.save_h5", True):
            print("\n[Export] Saving models...")
            hybrid.export_models()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
