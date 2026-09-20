"""K-Fold cross validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.config import Config
from utils.dataset import DatasetManager, SampleRecord
from utils.metrics import MetricsCalculator


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""

    fold: int
    metrics: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)


class CrossValidator:
    """K-Fold cross validation for the hybrid model."""

    def __init__(self, config: Config, n_folds: int = 5) -> None:
        self.config = config
        self.n_folds = n_folds
        self.dataset_manager = DatasetManager(config)
        self.metrics_calculator = MetricsCalculator(config.class_names)

    def run(
        self,
        train_fn: Callable,
        samples: Optional[List[SampleRecord]] = None,
    ) -> List[FoldResult]:
        """
        Run K-Fold cross validation.

        Args:
            train_fn: Callable accepting (train_samples, val_samples, fold_idx) and returning (model, history, metrics).
            samples: Optional pre-loaded samples.

        Returns:
            List of FoldResult for each fold.
        """
        if samples is None:
            self.dataset_manager.load()
            samples = self.dataset_manager.samples

        labels = [s.label for s in samples]
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.config.get("project.seed", 42),
        )

        results: List[FoldResult] = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"\n=== Fold {fold_idx + 1}/{self.n_folds} ===")
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            _, history, metrics = train_fn(train_samples, val_samples, fold_idx)
            results.append(FoldResult(fold=fold_idx + 1, metrics=metrics, history=history))

        self._save_summary(results)
        return results

    def _save_summary(self, results: List[FoldResult]) -> None:
        """Save cross-validation summary to file."""
        summary_path = self.config.output_dir / "reports" / "cross_validation_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        metric_keys = results[0].metrics.keys() if results else []
        with open(summary_path, "w", encoding="utf-8") as file:
            for result in results:
                file.write(f"Fold {result.fold}:\n")
                for key in metric_keys:
                    file.write(f"  {key}: {result.metrics.get(key, 0.0):.4f}\n")
                file.write("\n")

            file.write("Mean ± Std:\n")
            for key in metric_keys:
                values = [r.metrics.get(key, 0.0) for r in results]
                file.write(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")

        print(f"Cross-validation summary saved to {summary_path}")
