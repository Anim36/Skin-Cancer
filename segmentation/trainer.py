"""U-Net training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model

from segmentation.unet import UNet
from utils.callbacks import build_callbacks, enable_mixed_precision
from utils.config import Config
from utils.dataset import DatasetManager, SampleRecord
from utils.optimizers import get_optimizer


class SegmentationTrainer:
    """Train U-Net for lesion segmentation."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.input_size = tuple(config.get("segmentation.input_size", [256, 256]))
        self.checkpoint_path = config.checkpoint_dir / config.get("segmentation.checkpoint_name", "unet_best.keras")

    def build_model(self) -> Model:
        """Build U-Net model."""
        unet = UNet(
            input_size=self.input_size,
            filters=self.config.get("segmentation.filters", [64, 128, 256, 512]),
            dropout=self.config.get("segmentation.dropout", 0.3),
        )
        return unet.build()

    def compile_model(self, model: Model) -> Model:
        """Compile U-Net with optimizer and loss."""
        optimizer = get_optimizer(
            self.config.get("training.optimizer", "adam"),
            learning_rate=self.config.get("segmentation.learning_rate", 1e-4),
        )
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2, name="iou")],
        )
        return model

    def train(
        self,
        train_samples: Optional[List[SampleRecord]] = None,
        val_samples: Optional[List[SampleRecord]] = None,
        resume: bool = True,
    ) -> Tuple[Model, Dict[str, List[float]]]:
        """
        Train U-Net segmentation model.

        Returns:
            Tuple of (trained model, training history).
        """
        enable_mixed_precision(self.config.get("training.mixed_precision", True))

        if train_samples is None or val_samples is None:
            self.dataset_manager.load()
            train_samples, val_samples, _ = self.dataset_manager.split_data()

        batch_size = self.config.get("segmentation.batch_size", 8)
        train_ds = self.dataset_manager.create_tf_dataset(
            train_samples, batch_size=batch_size, shuffle=True, for_segmentation=True
        )
        val_ds = self.dataset_manager.create_tf_dataset(
            val_samples, batch_size=batch_size, shuffle=False, for_segmentation=True
        )

        model = self.build_model()
        if resume and self.checkpoint_path.exists():
            model = tf.keras.models.load_model(self.checkpoint_path, compile=False)
            print(f"Loaded segmentation checkpoint: {self.checkpoint_path}")

        model = self.compile_model(model)

        callbacks = build_callbacks(
            checkpoint_path=self.checkpoint_path,
            log_dir=self.config.log_dir / "segmentation",
            early_stopping_patience=self.config.get("training.early_stopping_patience", 15),
            reduce_lr_patience=self.config.get("training.reduce_lr_patience", 5),
            reduce_lr_factor=self.config.get("training.reduce_lr_factor", 0.5),
            min_lr=self.config.get("training.min_lr", 1e-7),
            monitor="val_loss",
            resume_path=self.checkpoint_path if resume else None,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.get("segmentation.epochs", 50),
            callbacks=callbacks,
        )
        return model, history.history

    def predict_masks(self, model: Model, samples: List[SampleRecord]) -> List[tf.Tensor]:
        """Generate lesion masks for given samples."""
        ds = self.dataset_manager.create_tf_dataset(
            samples,
            batch_size=self.config.get("segmentation.batch_size", 8),
            shuffle=False,
            for_segmentation=True,
        )
        masks = model.predict(ds)
        return [masks[i] for i in range(len(masks))]
