"""End-to-end hybrid model orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from classification.efficientnet_model import HybridClassifier
from explainability.gradcam import GradCAM
from attention.cbam import CBAM
from attention.se_block import SEBlock
from classification.metadata_fusion import MetadataFusion
from segmentation.trainer import SegmentationTrainer
from segmentation.unet import UNet
from utils.callbacks import build_callbacks, enable_mixed_precision
from utils.config import Config
from utils.dataset import DatasetManager, SampleRecord
from utils.export import ModelExporter
from utils.losses import get_loss
from utils.metrics import MetricsCalculator
from utils.optimizers import get_optimizer
from utils.visualization import Visualizer


class HybridSkinCancerModel:
    """
    Orchestrates segmentation, classification, training, evaluation, and explainability.

    Pipeline:
        Input -> Preprocessing -> U-Net Segmentation -> EfficientNetV2 + Metadata Fusion
        -> Attention -> Classification -> Grad-CAM
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.segmentation_trainer = SegmentationTrainer(config)
        self.classifier_builder = HybridClassifier(config)
        self.metrics_calculator = MetricsCalculator(config.class_names)
        self.visualizer = Visualizer(config.output_dir, config.class_names)
        self.exporter = ModelExporter(config.output_dir / "exports")

        self.segmentation_model: Optional[Model] = None
        self.classification_model: Optional[Model] = None
        self.gradcam: Optional[GradCAM] = None

        self.seg_checkpoint = config.checkpoint_dir / config.get("segmentation.checkpoint_name", "unet_best.keras")
        self.cls_checkpoint = config.checkpoint_dir / config.get("classification.checkpoint_name", "classifier_best.keras")

    def load_data(self) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
        """Load dataset and split into train/val/test."""
        if self.config.get("download.auto_download", False):
            from utils.download import DatasetDownloader

            DatasetDownloader(self.config).download()

        self.dataset_manager.load()
        return self.dataset_manager.split_data()

    def train_segmentation(
        self,
        train_samples: Optional[List[SampleRecord]] = None,
        val_samples: Optional[List[SampleRecord]] = None,
    ) -> Tuple[Model, Dict]:
        """Train U-Net segmentation model."""
        if not self.config.get("segmentation.enabled", True):
            print("Segmentation disabled in config.")
            return None, {}

        model, history = self.segmentation_trainer.train(train_samples, val_samples)
        self.segmentation_model = model
        return model, history

    def load_segmentation(self) -> Optional[Model]:
        """Load pretrained segmentation model if available."""
        if not self.config.get("segmentation.enabled", True):
            return None
        if self.seg_checkpoint.exists():
            self.segmentation_model = tf.keras.models.load_model(self.seg_checkpoint, compile=False)
            print(f"Loaded segmentation model: {self.seg_checkpoint}")
        return self.segmentation_model

    def build_classifier(self) -> Model:
        """Build hybrid classification model."""
        if self.segmentation_model is None:
            self.load_segmentation()
        self.classification_model = self.classifier_builder.build(self.segmentation_model)
        return self.classification_model

    def compile_classifier(self, model: Model, learning_rate: Optional[float] = None) -> Model:
        """Compile classifier with configured loss and optimizer."""
        lr = learning_rate or self.config.get("classification.learning_rate", 1e-4)
        optimizer = get_optimizer(
            self.config.get("training.optimizer", "adam"),
            learning_rate=lr,
            weight_decay=self.config.get("training.weight_decay", 1e-4),
        )
        loss = get_loss(
            self.config.get("training.loss", "categorical_crossentropy"),
            focal_gamma=self.config.get("training.focal_gamma", 2.0),
            focal_alpha=self.config.get("training.focal_alpha", 0.25),
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc", multi_label=False),
            ],
        )
        return model

    def train_classifier(
        self,
        train_samples: Optional[List[SampleRecord]] = None,
        val_samples: Optional[List[SampleRecord]] = None,
        fine_tune: bool = True,
    ) -> Tuple[Model, Dict]:
        """Train hybrid classifier with optional fine-tuning phase."""
        enable_mixed_precision(self.config.get("training.mixed_precision", True))

        if train_samples is None or val_samples is None:
            train_samples, val_samples, _ = self.load_data()

        batch_size = self.config.get("classification.batch_size", 16)
        train_ds = self.dataset_manager.create_tf_dataset(
            train_samples, batch_size=batch_size, shuffle=True, augment=True
        )
        val_ds = self.dataset_manager.create_tf_dataset(
            val_samples, batch_size=batch_size, shuffle=False, augment=False
        )

        resume = self.config.get("training.resume", True)
        if resume and self.cls_checkpoint.exists():
            self.classification_model = tf.keras.models.load_model(
                self.cls_checkpoint,
                custom_objects={"CBAM": CBAM, "SEBlock": SEBlock, "MetadataFusion": MetadataFusion},
                compile=False,
            )
            print(f"Resumed classifier from {self.cls_checkpoint}")
        else:
            self.build_classifier()

        self.compile_classifier(self.classification_model)
        class_weights = self.dataset_manager.get_class_weights()

        callbacks = build_callbacks(
            checkpoint_path=self.cls_checkpoint,
            log_dir=self.config.log_dir / "classification",
            early_stopping_patience=self.config.get("training.early_stopping_patience", 15),
            reduce_lr_patience=self.config.get("training.reduce_lr_patience", 5),
            reduce_lr_factor=self.config.get("training.reduce_lr_factor", 0.5),
            min_lr=self.config.get("training.min_lr", 1e-7),
            monitor="val_loss",
            resume_path=self.cls_checkpoint if resume else None,
        )

        history = self.classification_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.get("classification.epochs", 100),
            class_weight=class_weights,
            callbacks=callbacks,
        )

        if fine_tune:
            self.classifier_builder.fine_tune(self.classification_model)
            self.compile_classifier(
                self.classification_model,
                learning_rate=self.config.get("classification.fine_tune_learning_rate", 1e-5),
            )
            fine_tune_epochs = max(10, self.config.get("classification.epochs", 100) // 5)
            ft_history = self.classification_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
            )
            for key, values in ft_history.history.items():
                history.history.setdefault(key, []).extend(values)

        self.visualizer.plot_training_history(history.history, prefix="classification")
        self.gradcam = GradCAM(self.classification_model)
        return self.classification_model, history.history

    def evaluate(
        self,
        test_samples: Optional[List[SampleRecord]] = None,
    ) -> Dict:
        """Evaluate classifier on test set."""
        if test_samples is None:
            _, _, test_samples = self.load_data()

        if self.classification_model is None:
            if self.cls_checkpoint.exists():
                self.classification_model = tf.keras.models.load_model(self.cls_checkpoint, compile=False)
            else:
                raise FileNotFoundError("No trained classifier found.")

        batch_size = self.config.get("classification.batch_size", 16)
        test_ds = self.dataset_manager.create_tf_dataset(test_samples, batch_size=batch_size, shuffle=False)

        y_prob = self.classification_model.predict(test_ds)
        y_pred = np.argmax(y_prob, axis=-1)

        y_true = []
        for batch in test_ds:
            labels = batch[1].numpy()
            y_true.extend(np.argmax(labels, axis=-1))
        y_true = np.array(y_true)

        result = self.metrics_calculator.compute(y_true, y_pred, y_prob)
        metrics_dict = result.to_dict()
        metrics_dict.update(result.per_class_accuracy)

        print("\n=== Evaluation Results ===")
        print(result.classification_report)
        print(f"Accuracy: {result.accuracy:.4f}, F1: {result.f1:.4f}, AUC: {result.auc:.4f}")

        self.visualizer.plot_confusion_matrix(result.confusion)
        self.visualizer.plot_roc_curve(y_true, y_prob)
        self.visualizer.save_metrics_report(metrics_dict, "evaluation_metrics.txt")

        report_path = self.config.output_dir / "reports" / "classification_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(result.classification_report, encoding="utf-8")

        if self.config.get("evaluation.save_gradcam", True):
            self._generate_test_gradcam(test_samples[:5])

        return metrics_dict

    def _generate_test_gradcam(self, samples: List[SampleRecord]) -> None:
        """Generate Grad-CAM for sample test images."""
        if self.gradcam is None:
            self.gradcam = GradCAM(self.classification_model)

        for sample in samples:
            image = self.dataset_manager.preprocessor.process(
                __import__("cv2").imread(str(sample.image_path))[:, :, ::-1]
            )
            metadata = np.array([[sample.age, sample.gender, sample.location]], dtype=np.float32)
            output_path = self.config.output_dir / "gradcam" / f"{sample.image_id}_gradcam.png"
            self.gradcam.generate_and_save(
                np.expand_dims(image, 0),
                metadata,
                output_path,
                self.config.class_names,
                visualizer=self.visualizer,
            )

    def predict(
        self,
        image: np.ndarray,
        age: float,
        gender: str,
        location: str,
    ) -> Dict:
        """
        Run inference on a single image with clinical metadata.

        Returns:
            Dictionary with prediction, confidence, risk level, and heatmap path.
        """
        if self.classification_model is None:
            if self.cls_checkpoint.exists():
                self.classification_model = tf.keras.models.load_model(self.cls_checkpoint, compile=False)
            else:
                raise FileNotFoundError("No trained model available for prediction.")

        gender_classes = self.config.get("metadata.gender_classes", ["male", "female", "unknown"])
        location_classes = self.config.get("metadata.location_classes", [])
        gender_idx = gender_classes.index(gender.lower()) if gender.lower() in gender_classes else 2
        location_idx = location_classes.index(location.lower()) if location.lower() in location_classes else len(location_classes) - 1
        age_norm = np.clip(age, 0, 100) / 100.0

        processed = self.dataset_manager.preprocessor.process(image)
        metadata = np.array([[age_norm, gender_idx, location_idx]], dtype=np.float32)
        image_batch = np.expand_dims(processed, axis=0)

        probs = self.classification_model.predict([image_batch, metadata], verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        disease = self.config.class_names[pred_idx]

        risk_level = self._compute_risk_level(disease, confidence)

        heatmap_path = None
        if self.config.get("evaluation.save_gradcam", True):
            if self.gradcam is None:
                self.gradcam = GradCAM(self.classification_model)
            output_path = self.config.output_dir / "gradcam" / "inference_gradcam.png"
            self.gradcam.generate_and_save(
                image_batch,
                metadata,
                output_path,
                self.config.class_names,
                original_image=image,
                visualizer=self.visualizer,
            )
            heatmap_path = str(output_path)

        return {
            "disease": disease,
            "confidence": confidence,
            "probabilities": {self.config.class_names[i]: float(probs[i]) for i in range(len(probs))},
            "risk_level": risk_level,
            "gradcam_path": heatmap_path,
        }

    @staticmethod
    def _compute_risk_level(disease: str, confidence: float) -> str:
        """Map prediction to clinical risk level."""
        high_risk = {"mel", "bcc", "akiec", "scc"}
        medium_risk = {"bkl", "df", "vasc"}
        if disease.lower() in high_risk:
            if confidence >= 0.7:
                return "HIGH - Consult dermatologist immediately"
            return "MODERATE-HIGH - Further examination recommended"
        if disease.lower() in medium_risk:
            return "MODERATE - Monitor and follow up"
        return "LOW - Likely benign, routine check advised"

    def export_models(self) -> Dict:
        """Export trained models to H5, ONNX, and TFLite."""
        if self.classification_model is None and self.cls_checkpoint.exists():
            self.classification_model = tf.keras.models.load_model(self.cls_checkpoint, compile=False)
        if self.classification_model is None:
            raise FileNotFoundError("No model to export.")
        return self.exporter.export_all(self.classification_model, prefix="classifier")

    def compare_losses_and_optimizers(self) -> Dict:
        """Run quick comparison of loss functions and optimizers (documentation utility)."""
        from utils.losses import compare_losses
        from utils.optimizers import compare_optimizers

        dummy_true = tf.one_hot([0, 1, 2], depth=self.config.num_classes)
        dummy_pred = tf.nn.softmax(tf.random.normal((3, self.config.num_classes)))
        loss_comparison = compare_losses(dummy_true, dummy_pred)
        optimizer_comparison = {name: type(opt).__name__ for name, opt in compare_optimizers().items()}
        return {"losses": loss_comparison, "optimizers": optimizer_comparison}
