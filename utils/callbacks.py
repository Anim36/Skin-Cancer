"""Training callbacks for model checkpointing and monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


class ResumeCheckpointCallback(Callback):
    """Callback to log checkpoint resume information."""

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def on_train_begin(self, logs=None):
        if self.checkpoint_path and self.checkpoint_path.exists():
            print(f"Resuming training from checkpoint: {self.checkpoint_path}")


def build_callbacks(
    checkpoint_path: Path,
    log_dir: Path,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    monitor: str = "val_loss",
    resume_path: Optional[Path] = None,
) -> List[Callback]:
    """
    Build standard training callbacks.

    Args:
        checkpoint_path: Path to save best model weights.
        log_dir: TensorBoard log directory.
        early_stopping_patience: Patience for early stopping.
        reduce_lr_patience: Patience for learning rate reduction.
        reduce_lr_factor: Factor to reduce learning rate.
        min_lr: Minimum learning rate.
        monitor: Metric to monitor.
        resume_path: Optional checkpoint used for resume logging.

    Returns:
        List of Keras callbacks.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks: List[Callback] = [
        EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        TensorBoard(log_dir=str(log_dir), histogram_freq=0, write_graph=True),
    ]

    if resume_path is not None:
        callbacks.insert(0, ResumeCheckpointCallback(resume_path))

    return callbacks


def enable_mixed_precision(enabled: bool = True) -> None:
    """Enable mixed precision training policy."""
    if enabled:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled.")
    else:
        policy = tf.keras.mixed_precision.Policy("float32")
        tf.keras.mixed_precision.set_global_policy(policy)
