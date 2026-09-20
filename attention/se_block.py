"""Squeeze-and-Excitation (SE) attention block."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, ratio: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = tf.keras.Sequential(
            [
                layers.Dense(max(channels // self.ratio, 1), activation="relu"),
                layers.Dense(channels, activation="sigmoid"),
            ]
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        scale = self.squeeze(inputs)
        scale = self.excitation(scale)
        scale = tf.reshape(scale, (-1, 1, 1, tf.shape(inputs)[-1]))
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config
