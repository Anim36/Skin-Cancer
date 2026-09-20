"""Clinical metadata embedding and fusion."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class MetadataFusion(layers.Layer):
    """Embed and fuse clinical metadata with image features."""

    def __init__(
        self,
        metadata_dim: int = 32,
        num_genders: int = 3,
        num_locations: int = 15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.metadata_dim = metadata_dim
        self.num_genders = num_genders
        self.num_locations = num_locations

        self.age_dense = layers.Dense(metadata_dim, activation="relu", name="age_embedding")
        self.gender_embedding = layers.Embedding(num_genders, metadata_dim, name="gender_embedding")
        self.location_embedding = layers.Embedding(num_locations, metadata_dim, name="location_embedding")
        self.fusion_dense = layers.Dense(metadata_dim, activation="relu", name="metadata_fusion")
        self.batch_norm = layers.BatchNormalization(name="metadata_bn")

    def call(self, metadata: tf.Tensor, training=False) -> tf.Tensor:
        """
        Process metadata tensor [age, gender_idx, location_idx].

        Args:
            metadata: Float tensor of shape (batch, 3).

        Returns:
            Metadata embedding of shape (batch, metadata_dim).
        """
        age = metadata[:, 0:1]
        gender = tf.cast(metadata[:, 1], tf.int32)
        location = tf.cast(metadata[:, 2], tf.int32)

        age_emb = self.age_dense(age)
        gender_emb = self.gender_embedding(gender)
        location_emb = self.location_embedding(location)

        combined = tf.concat([age_emb, gender_emb, location_emb], axis=-1)
        fused = self.fusion_dense(combined)
        return self.batch_norm(fused, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "metadata_dim": self.metadata_dim,
                "num_genders": self.num_genders,
                "num_locations": self.num_locations,
            }
        )
        return config
