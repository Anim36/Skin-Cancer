"""Shared model loading utilities."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model

from attention.cbam import CBAM
from attention.se_block import SEBlock
from classification.metadata_fusion import MetadataFusion

CUSTOM_OBJECTS = {
    "CBAM": CBAM,
    "SEBlock": SEBlock,
    "MetadataFusion": MetadataFusion,
}


def load_keras_model(path: str, compile: bool = False) -> Model:
    """Load Keras model with project custom layers."""
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=compile)
