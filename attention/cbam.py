"""Convolutional Block Attention Module (CBAM)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    """Channel attention sub-module."""

    def __init__(self, ratio: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense_one = layers.Dense(max(channels // self.ratio, 1), activation="relu")
        self.shared_dense_two = layers.Dense(channels)
        super().build(input_shape)

    def call(self, inputs, training=False):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_two(self.shared_dense_one(tf.squeeze(avg_pool, axis=[1, 2])))
        max_out = self.shared_dense_two(self.shared_dense_one(tf.squeeze(max_pool, axis=[1, 2])))
        attention = tf.nn.sigmoid(avg_out + max_out)
        attention = tf.reshape(attention, (-1, 1, 1, tf.shape(inputs)[-1]))
        return inputs * attention


class SpatialAttention(layers.Layer):
    """Spatial attention sub-module."""

    def __init__(self, kernel_size: int = 7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding="same", activation="sigmoid")

    def call(self, inputs, training=False):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention


class CBAM(layers.Layer):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, ratio: int = 8, kernel_size: int = 7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, inputs, training=False):
        x = self.channel_attention(inputs, training=training)
        x = self.spatial_attention(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        return config
