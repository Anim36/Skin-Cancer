"""Optimizer factory for training."""

from __future__ import annotations

from typing import Union

import tensorflow as tf
from tensorflow.keras import optimizers


def get_optimizer(
    optimizer_name: str = "adam",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> optimizers.Optimizer:
    """
    Create Adam or AdamW optimizer.

    Args:
        optimizer_name: 'adam' or 'adamw'.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW.

    Returns:
        Configured Keras optimizer.
    """
    name = optimizer_name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def compare_optimizers(
    optimizer_names: list[str] | None = None,
    learning_rate: float = 1e-4,
) -> dict[str, optimizers.Optimizer]:
    """Return dictionary of optimizers for comparison experiments."""
    if optimizer_names is None:
        optimizer_names = ["adam", "adamw"]
    return {name: get_optimizer(name, learning_rate=learning_rate) for name in optimizer_names}
