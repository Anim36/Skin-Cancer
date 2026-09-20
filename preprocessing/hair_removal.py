"""Hair removal using morphological operations."""

from __future__ import annotations

from typing import Tuple, Union

import cv2
import numpy as np


class HairRemover:
    """Remove hair artifacts from dermoscopic images using morphological closing."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (17, 17),
        threshold: int = 10,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    def remove(self, image: np.ndarray) -> np.ndarray:
        """
        Apply hair removal to RGB image.

        Args:
            image: Input BGR or RGB image (uint8 or float).

        Returns:
            Inpainted image with reduced hair artifacts.
        """
        if image.dtype != np.uint8:
            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image.copy()

        if len(image_uint8.shape) == 2:
            gray = image_uint8
            color = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        else:
            color = image_uint8
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        _, hair_mask = cv2.threshold(blackhat, self.threshold, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(color, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        if len(image.shape) == 2:
            return cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    def remove_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply hair removal to a batch of images."""
        return np.stack([self.remove(img) for img in images], axis=0)
