"""Visualization utilities for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sk_auc


class Visualizer:
    """Generate plots for model training and evaluation."""

    def __init__(self, output_dir: Path, class_names: List[str]) -> None:
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        sns.set_style("whitegrid")

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        prefix: str = "training",
    ) -> Tuple[Path, Path]:
        """Plot training/validation loss and accuracy curves."""
        loss_path = self.plots_dir / f"{prefix}_loss.png"
        acc_path = self.plots_dir / f"{prefix}_accuracy.png"

        epochs = range(1, len(history.get("loss", [])) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history.get("loss", []), label="Training Loss")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()

        if "accuracy" in history or "val_accuracy" in history:
            plt.figure(figsize=(10, 5))
            if "accuracy" in history:
                plt.plot(epochs, history["accuracy"], label="Training Accuracy")
            if "val_accuracy" in history:
                plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(acc_path, dpi=150)
            plt.close()

        return loss_path, acc_path

    def plot_confusion_matrix(
        self,
        confusion: np.ndarray,
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """Plot confusion matrix heatmap."""
        output_path = self.plots_dir / filename
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        filename: str = "roc_curve.png",
    ) -> Path:
        """Plot ROC curve for binary or multi-class (OvR) settings."""
        output_path = self.plots_dir / filename
        plt.figure(figsize=(8, 6))

        if y_prob.ndim == 1 or y_prob.shape[-1] == 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = sk_auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        else:
            for idx, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == idx).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, idx])
                roc_auc = sk_auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def plot_gradcam(
        self,
        original: np.ndarray,
        heatmap: np.ndarray,
        filename: str = "gradcam.png",
        title: Optional[str] = None,
    ) -> Path:
        """Overlay Grad-CAM heatmap on original image."""
        output_path = self.output_dir / "gradcam" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)

        heatmap_resized = np.uint8(255 * heatmap)
        import cv2

        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def save_metrics_report(self, metrics: Dict[str, float], filename: str = "metrics.txt") -> Path:
        """Save metrics dictionary to text file."""
        report_path = self.output_dir / "reports" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as file:
            for key, value in metrics.items():
                file.write(f"{key}: {value:.6f}\n")
        return report_path
