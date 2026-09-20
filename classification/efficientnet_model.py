"""Hybrid EfficientNetV2 classifier with metadata fusion and attention."""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers

from attention.cbam import CBAM
from attention.se_block import SEBlock
from classification.metadata_fusion import MetadataFusion
from utils.config import Config


class HybridClassifier:
    """
    EfficientNetV2-based classifier with clinical metadata fusion and attention.

    Pipeline:
        Image -> EfficientNetV2 -> Attention -> Global Pool -> Fusion with Metadata -> Classification
    """

    BACKBONE_MAP = {
        "efficientnetv2-b0": "tf.keras.applications.EfficientNetV2B0",
        "efficientnetv2-b1": "EfficientNetV2B1",
        "efficientnetv2-s": "EfficientNetV2S",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.input_size = tuple(config.get("classification.input_size", [224, 224]))
        self.num_classes = config.num_classes
        self.backbone_name = config.get("classification.backbone", "efficientnetv2-b0").lower()
        self.attention_type = config.get("classification.attention_type", "cbam").lower()
        self.dropout = config.get("classification.dropout", 0.4)
        self.metadata_dim = config.get("classification.metadata_dim", 32)

        gender_classes = config.get("metadata.gender_classes", ["male", "female", "unknown"])
        location_classes = config.get("metadata.location_classes", [])
        self.num_genders = len(gender_classes)
        self.num_locations = len(location_classes)

    def _get_backbone(self, input_tensor: tf.Tensor) -> Tuple[tf.Tensor, Model]:
        """Load EfficientNetV2 backbone with transfer learning."""
        weights = "imagenet"
        if self.backbone_name == "efficientnetv2-b0":
            base = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor,
                pooling=None,
            )
        elif self.backbone_name == "efficientnetv2-b1":
            base = tf.keras.applications.EfficientNetV2B1(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor,
                pooling=None,
            )
        elif self.backbone_name == "efficientnetv2-s":
            base = tf.keras.applications.EfficientNetV2S(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor,
                pooling=None,
            )
        else:
            base = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor,
                pooling=None,
            )

        if self.config.get("classification.freeze_base", True):
            base.trainable = False

        return base.output, base

    def _apply_attention(self, x: tf.Tensor) -> tf.Tensor:
        """Apply CBAM, SE, or no attention."""
        if self.attention_type == "cbam":
            return CBAM(name="cbam_attention")(x)
        if self.attention_type == "se":
            return SEBlock(name="se_attention")(x)
        return x

    def build(self, segmentation_model: Optional[Model] = None) -> Model:
        """
        Build hybrid classification model.

        Args:
            segmentation_model: Optional U-Net for lesion masking.

        Returns:
            Keras Model accepting (image, metadata) inputs.
        """
        image_input = layers.Input(shape=(*self.input_size, 3), name="image_input")
        metadata_input = layers.Input(shape=(3,), name="metadata_input", dtype=tf.float32)

        x = image_input
        if segmentation_model is not None and self.config.get("segmentation.enabled", True):
            mask = segmentation_model(image_input)
            from segmentation.unet import UNet

            x = UNet.apply_mask(image_input, mask)

        features, backbone = self._get_backbone(x)
        features = self._apply_attention(features)
        features = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)

        metadata_features = MetadataFusion(
            metadata_dim=self.metadata_dim,
            num_genders=self.num_genders,
            num_locations=self.num_locations,
            name="metadata_fusion",
        )(metadata_input)

        fused = layers.Concatenate(name="feature_fusion")([features, metadata_features])
        x = layers.Dense(256, activation="relu", name="fc1")(fused)
        x = layers.BatchNormalization(name="fc1_bn")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(128, activation="relu", name="fc2")(x)
        x = layers.Dropout(self.dropout / 2)(x)

        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            dtype="float32",
            name="classification_output",
        )(x)

        model = Model(inputs=[image_input, metadata_input], outputs=outputs, name="HybridSkinCancerClassifier")
        model.backbone = backbone  # type: ignore[attr-defined]
        return model

    def fine_tune(self, model: Model, fine_tune_at: Optional[int] = None) -> Model:
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            model: Trained model with frozen backbone.
            fine_tune_at: Layer index to start fine-tuning from.

        Returns:
            Model ready for fine-tuning.
        """
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            for layer in model.layers:
                if "efficientnet" in layer.name.lower():
                    backbone = layer
                    break

        if backbone is None:
            return model

        fine_tune_at = fine_tune_at or self.config.get("classification.fine_tune_at", 100)
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False

        print(f"Fine-tuning from layer {fine_tune_at}. Trainable layers: {sum(l.trainable for l in model.layers)}")
        return model
