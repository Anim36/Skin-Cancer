"""U-Net architecture for lesion segmentation."""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers


class UNet:
    """U-Net model builder for skin lesion segmentation."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        filters: List[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        self.input_size = input_size
        self.filters = filters or [64, 128, 256, 512]
        self.dropout = dropout

    def _conv_block(self, inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(inputs)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return x

    def build(self) -> Model:
        """Build and compile U-Net model."""
        inputs = layers.Input(shape=(*self.input_size, 3), name="unet_input")

        # Encoder
        skip_connections = []
        x = inputs
        for idx, f in enumerate(self.filters):
            x = self._conv_block(x, f, f"enc_{idx}")
            skip_connections.append(x)
            x = layers.MaxPooling2D(2, name=f"pool_{idx}")(x)

        # Bottleneck
        x = self._conv_block(x, self.filters[-1] * 2, "bottleneck")
        x = layers.Dropout(self.dropout)(x)

        # Decoder
        for idx, f in enumerate(reversed(self.filters)):
            x = layers.UpSampling2D(2, name=f"upsample_{idx}")(x)
            skip = skip_connections[-(idx + 1)]
            x = layers.Concatenate(name=f"concat_{idx}")([x, skip])
            x = self._conv_block(x, f, f"dec_{idx}")

        outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask_output")(x)
        return Model(inputs=inputs, outputs=outputs, name="UNet")

    @staticmethod
    def apply_mask(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Apply segmentation mask to focus on lesion region."""
        if mask.shape[-1] != 1:
            mask = tf.reduce_mean(mask, axis=-1, keepdims=True)
        if mask.shape[1:3] != image.shape[1:3]:
            mask = tf.image.resize(mask, tf.shape(image)[1:3])
        return image * mask + image * (1.0 - mask) * 0.1
