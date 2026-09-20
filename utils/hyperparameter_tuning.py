"""Hyperparameter tuning with Optuna."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import optuna

from utils.config import Config


class HyperparameterTuner:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        config: Config,
        objective_fn: Callable[[optuna.Trial], float],
        study_name: Optional[str] = None,
        n_trials: int = 20,
    ) -> None:
        self.config = config
        self.objective_fn = objective_fn
        self.study_name = study_name or config.get("hyperparameter_tuning.study_name", "skin_cancer_optuna")
        self.n_trials = n_trials or config.get("hyperparameter_tuning.n_trials", 20)
        self.storage_path = config.output_dir / "optuna" / f"{self.study_name}.db"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> optuna.Study:
        """Execute hyperparameter search."""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.storage_path}",
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(self.objective_fn, n_trials=self.n_trials, show_progress_bar=True)
        self._save_results(study)
        return study

    def _save_results(self, study: optuna.Study) -> None:
        """Save best trial parameters and value."""
        results_path = self.config.output_dir / "reports" / "optuna_best_params.txt"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as file:
            file.write(f"Best value: {study.best_value:.6f}\n")
            file.write("Best params:\n")
            for key, value in study.best_params.items():
                file.write(f"  {key}: {value}\n")
        print(f"Optuna results saved to {results_path}")


def suggest_hyperparameters(trial: optuna.Trial, config: Config) -> Dict:
    """Suggest hyperparameters for Optuna trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "loss": trial.suggest_categorical("loss", ["categorical_crossentropy", "focal"]),
        "attention_type": trial.suggest_categorical("attention_type", ["cbam", "se", "none"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
