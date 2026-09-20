"""Image normalization utilities."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


class ImageNormalizer:
    """Normalize images using ImageNet, min-max, or z-score statistics."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        method: Literal["imagenet", "minmax", "zscore"] = "imagenet",
    ) -> None:
        self.method = method

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image array.

        Args:
            image: RGB image as float or uint8.

        Returns:
            Normalized float32 image.
        """
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        if self.method == "imagenet":
            return (image - self.IMAGENET_MEAN) / self.IMAGENET_STD
        if self.method == "minmax":
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val < 1e-7:
                return image
            return (image - min_val) / (max_val - min_val)
        if self.method == "zscore":
            mean = image.mean(axis=(0, 1), keepdims=True)
            std = image.std(axis=(0, 1), keepdims=True) + 1e-7
            return (image - mean) / std
        raise ValueError(f"Unknown normalization method: {self.method}")

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """Reverse ImageNet normalization for visualization."""
        if self.method == "imagenet":
            restored = image * self.IMAGENET_STD + self.IMAGENET_MEAN
            return np.clip(restored * 255, 0, 255).astype(np.uint8)
        if image.max() <= 1.0:
            return np.clip(image * 255, 0, 255).astype(np.uint8)
        return image.astype(np.uint8)

    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image using OpenCV."""
        import cv2

        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
