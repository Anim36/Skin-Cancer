"""Data augmentation pipeline using Albumentations."""

from __future__ import annotations

import albumentations as A
import numpy as np
import tensorflow as tf

from utils.config import Config


class AugmentationPipeline:
    """Configurable augmentation for training data."""

    def __init__(self, config: Config) -> None:
        self.enabled = config.get("augmentation.enabled", True)
        self.transform = A.Compose(
            [
                A.Rotate(
                    limit=config.get("augmentation.rotation_limit", 30),
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5 if config.get("augmentation.horizontal_flip", True) else 0.0),
                A.VerticalFlip(p=0.5 if config.get("augmentation.vertical_flip", True) else 0.0),
                A.RandomScale(scale_limit=config.get("augmentation.zoom_limit", 0.2), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config.get("augmentation.brightness_limit", 0.2),
                    contrast_limit=config.get("augmentation.contrast_limit", 0.2),
                    p=0.5,
                ),
            ]
        )

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single image."""
        if not self.enabled:
            return image
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        augmented = self.transform(image=image_uint8)["image"]
        return augmented.astype(np.float32) / 255.0

    def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Map augmentation onto TensorFlow dataset."""

        def _augment(inputs, labels):
            image, metadata = inputs

            def _aug_fn(img):
                return tf.py_function(
                    lambda x: self.augment(x.numpy()),
                    [img],
                    tf.float32,
                )

            image = _aug_fn(image)
            return (image, metadata), labels

        return dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
