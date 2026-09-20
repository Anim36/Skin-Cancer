"""End-to-end preprocessing pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from preprocessing.enhancement import CLAHEEnhancer, Denoiser
from preprocessing.hair_removal import HairRemover
from preprocessing.normalization import ImageNormalizer
from utils.config import Config


class PreprocessingPipeline:
    """Complete preprocessing: hair removal, CLAHE, denoising, resize, normalization."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.image_size = config.image_size
        self.apply_hair_removal = config.get("preprocessing.apply_hair_removal", True)
        self.apply_clahe = config.get("preprocessing.apply_clahe", True)
        self.apply_denoising = config.get("preprocessing.apply_denoising", True)

        self.hair_remover = HairRemover()
        self.clahe = CLAHEEnhancer(
            clip_limit=config.get("preprocessing.clahe_clip_limit", 2.0),
            tile_grid_size=tuple(config.get("preprocessing.clahe_tile_grid_size", [8, 8])),
        )
        self.denoiser = Denoiser(
            h=config.get("preprocessing.denoise_h", 10),
            template_window_size=config.get("preprocessing.denoise_template_window_size", 7),
            search_window_size=config.get("preprocessing.denoise_search_window_size", 21),
        )
        self.normalizer = ImageNormalizer(
            method=config.get("preprocessing.normalization", "imagenet"),
        )

    def process(self, image: np.ndarray, target_size: Tuple[int, int] | None = None) -> np.ndarray:
        """
        Apply full preprocessing pipeline to numpy image.

        Args:
            image: RGB image array.
            target_size: Optional override for output size.

        Returns:
            Preprocessed normalized image.
        """
        size = target_size or self.image_size

        if self.apply_hair_removal:
            image = self.hair_remover.remove(image)
        if self.apply_clahe:
            image = self.clahe.enhance(image)
        if self.apply_denoising:
            image = self.denoiser.denoise(image)

        image = self.normalizer.resize(image, (size[1], size[0]))
        return self.normalizer.normalize(image)

    def process_tensor(self, image: tf.Tensor, target_size: Tuple[int, int] | None = None) -> tf.Tensor:
        """Apply preprocessing pipeline within TensorFlow graph via py_function."""
        size = target_size or self.image_size

        def _process_fn(img: np.ndarray) -> np.ndarray:
            return self.process(img, target_size=size).astype(np.float32)

        processed = tf.py_function(_process_fn, [image], tf.float32)
        processed.set_shape((*size, 3))
        return processed

    def process_for_inference(self, image: np.ndarray) -> np.ndarray:
        """Process image for inference (same as process but explicit naming)."""
        return self.process(image)
