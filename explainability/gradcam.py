"""Grad-CAM and Grad-CAM++ for model explainability."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN explainability."""

    def __init__(
        self,
        model: Model,
        layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        self.grad_model = self._build_grad_model()

    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    if len(sublayer.output_shape) == 4:
                        return sublayer.name
        raise ValueError("No convolutional layer found for Grad-CAM.")

    def _build_grad_model(self) -> Model:
        """Build model that outputs conv feature maps and predictions."""
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output,
            ],
        )
        return grad_model

    def compute_heatmap(
        self,
        image: np.ndarray,
        metadata: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Compute Grad-CAM heatmap for a single prediction.

        Args:
            image: Preprocessed image array (1, H, W, 3).
            metadata: Metadata array (1, 3).
            class_idx: Target class index. Uses predicted class if None.

        Returns:
            Tuple of (heatmap, class_idx, prediction probabilities).
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        if metadata.ndim == 1:
            metadata = np.expand_dims(metadata, axis=0)

        inputs = [tf.constant(image, dtype=tf.float32), tf.constant(metadata, dtype=tf.float32)]

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
        return heatmap.numpy(), class_idx, predictions.numpy()[0]

    def generate_and_save(
        self,
        image: np.ndarray,
        metadata: np.ndarray,
        output_path: Path,
        class_names: List[str],
        original_image: Optional[np.ndarray] = None,
        visualizer=None,
    ) -> Path:
        """Generate heatmap and save visualization."""
        heatmap, class_idx, probs = self.compute_heatmap(image, metadata)
        title = f"Grad-CAM: {class_names[class_idx]} ({probs[class_idx]:.2%})"

        if visualizer is not None:
            display_image = original_image if original_image is not None else image[0] if image.ndim == 4 else image
            if display_image.max() <= 1.0 and display_image.min() >= -2:
                display_image = self._denormalize(display_image)
            return visualizer.plot_gradcam(display_image, heatmap, filename=output_path.name, title=title)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap="jet")
        plt.title(title)
        plt.axis("off")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def _denormalize(image: np.ndarray) -> np.ndarray:
        """Basic denormalization for display."""
        from preprocessing.normalization import ImageNormalizer

        if image.ndim == 4:
            image = image[0]
        return ImageNormalizer(method="imagenet").denormalize(image)


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ with improved localization for multiple object instances."""

    def compute_heatmap(
        self,
        image: np.ndarray,
        metadata: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        if metadata.ndim == 1:
            metadata = np.expand_dims(metadata, axis=0)

        inputs = [tf.constant(image, dtype=tf.float32), tf.constant(metadata, dtype=tf.float32)]

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        grads = grads[0]
        conv_outputs = conv_outputs[0]

        grad_squared = grads ** 2
        grad_cubed = grads ** 3
        sum_activations = tf.reduce_sum(conv_outputs, axis=(0, 1))
        alpha_denom = 2 * grad_squared + tf.reduce_sum(conv_outputs * grad_cubed, axis=(0, 1))
        alpha = grad_squared / (alpha_denom + 1e-7)
        weights = tf.reduce_sum(alpha * tf.maximum(grads, 0.0), axis=(0, 1))
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
        return heatmap.numpy(), class_idx, predictions.numpy()[0]
