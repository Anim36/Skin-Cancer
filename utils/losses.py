"""Loss functions for classification training."""

from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import losses


def focal_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Build focal loss for imbalanced classification.

    Args:
        gamma: Focusing parameter.
        alpha: Balancing factor.

    Returns:
        Callable loss function.
    """

    def _focal(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))

    return _focal


def get_loss(
    loss_name: str = "categorical_crossentropy",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> losses.Loss | Callable:
    """
    Return configured loss function.

    Args:
        loss_name: 'categorical_crossentropy' or 'focal'.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.

    Returns:
        Keras loss instance or callable.
    """
    loss_name = loss_name.lower()
    if loss_name == "focal":
        return focal_loss(gamma=focal_gamma, alpha=focal_alpha)
    if loss_name == "categorical_crossentropy":
        return losses.CategoricalCrossentropy()
    raise ValueError(f"Unsupported loss: {loss_name}")


def compare_losses(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> dict[str, float]:
    """Compare categorical crossentropy and focal loss values."""
    ce = losses.CategoricalCrossentropy()(y_true, y_pred).numpy()
    fl = focal_loss(gamma=focal_gamma, alpha=focal_alpha)(y_true, y_pred).numpy()
    return {"categorical_crossentropy": float(ce), "focal_loss": float(fl)}
