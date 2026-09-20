"""CLAHE enhancement and noise removal."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class CLAHEEnhancer:
    """Contrast Limited Adaptive Histogram Equalization for dermoscopic images."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to RGB image in LAB color space.

        Args:
            image: Input RGB image.

        Returns:
            CLAHE-enhanced RGB image.
        """
        if image.dtype != np.uint8:
            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image

        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


class Denoiser:
    """Non-local means denoising for skin lesion images."""

    def __init__(
        self,
        h: float = 10.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ) -> None:
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply fast non-local means denoising.

        Args:
            image: Input RGB image.

        Returns:
            Denoised RGB image.
        """
        if image.dtype != np.uint8:
            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image

        denoised = cv2.fastNlMeansDenoisingColored(
            image_uint8,
            None,
            self.h,
            self.h,
            self.template_window_size,
            self.search_window_size,
        )
        return denoised
