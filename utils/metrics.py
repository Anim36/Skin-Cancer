"""Evaluation metrics for skin cancer classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_report: str = ""
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    roc_curve: Optional[tuple] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert scalar metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "sensitivity": self.sensitivity,
            "f1": self.f1,
            "auc": self.auc,
        }


class MetricsCalculator:
    """Compute comprehensive classification metrics."""

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels (integer or one-hot).
            y_pred: Predicted labels (integer or one-hot).
            y_prob: Predicted probabilities for ROC/AUC.

        Returns:
            MetricsResult with all computed metrics.
        """
        y_true_int = self._to_int_labels(y_true)
        y_pred_int = self._to_int_labels(y_pred)

        cm = confusion_matrix(y_true_int, y_pred_int, labels=list(range(self.num_classes)))
        specificity = self._macro_specificity(cm)
        sensitivity = recall_score(y_true_int, y_pred_int, average="macro", zero_division=0)

        result = MetricsResult(
            accuracy=float(accuracy_score(y_true_int, y_pred_int)),
            precision=float(precision_score(y_true_int, y_pred_int, average="macro", zero_division=0)),
            recall=float(recall_score(y_true_int, y_pred_int, average="macro", zero_division=0)),
            specificity=float(specificity),
            sensitivity=float(sensitivity),
            f1=float(f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)),
            confusion=cm,
            classification_report=classification_report(
                y_true_int,
                y_pred_int,
                target_names=self.class_names,
                zero_division=0,
            ),
            per_class_accuracy=self._per_class_accuracy(cm),
        )

        if y_prob is not None:
            try:
                if y_prob.ndim == 1:
                    y_prob = np.expand_dims(y_prob, axis=-1)
                if self.num_classes == 2 and y_prob.shape[-1] == 1:
                    result.auc = float(roc_auc_score(y_true_int, y_prob))
                    fpr, tpr, _ = roc_curve(y_true_int, y_prob)
                else:
                    y_true_oh = self._to_one_hot(y_true_int)
                    result.auc = float(
                        roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
                    )
                    fpr, tpr, _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
                result.roc_curve = (fpr, tpr)
            except ValueError:
                result.auc = 0.0

        return result

    def _to_int_labels(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim > 1 and labels.shape[-1] > 1:
            return np.argmax(labels, axis=-1)
        return labels.astype(int)

    def _to_one_hot(self, labels: np.ndarray) -> np.ndarray:
        labels = self._to_int_labels(labels)
        one_hot = np.zeros((len(labels), self.num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return one_hot

    @staticmethod
    def _macro_specificity(confusion: np.ndarray) -> float:
        specificities = []
        for i in range(confusion.shape[0]):
            tp = confusion[i, i]
            fn = confusion[i, :].sum() - tp
            fp = confusion[:, i].sum() - tp
            tn = confusion.sum() - tp - fn - fp
            denom = tn + fp
            specificities.append(tn / denom if denom > 0 else 0.0)
        return float(np.mean(specificities))

    def _per_class_accuracy(self, confusion: np.ndarray) -> Dict[str, float]:
        per_class = {}
        for idx, name in enumerate(self.class_names):
            total = confusion[idx].sum()
            per_class[name] = float(confusion[idx, idx] / total) if total > 0 else 0.0
        return per_class
