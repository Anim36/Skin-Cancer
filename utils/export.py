"""Model export utilities for H5, ONNX, and TensorFlow Lite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tensorflow as tf


class ModelExporter:
    """Export trained models to multiple formats."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_h5(self, model: tf.keras.Model, filename: str = "model_best.h5") -> Path:
        """Export model to HDF5 format."""
        output_path = self.output_dir / filename
        model.save(output_path, save_format="h5")
        print(f"Model exported to H5: {output_path}")
        return output_path

    def export_keras(self, model: tf.keras.Model, filename: str = "model_best.keras") -> Path:
        """Export model to native Keras format."""
        output_path = self.output_dir / filename
        model.save(output_path)
        print(f"Model exported to Keras: {output_path}")
        return output_path

    def export_onnx(
        self,
        model: tf.keras.Model,
        filename: str = "model_best.onnx",
        input_signature: Optional[list] = None,
    ) -> Optional[Path]:
        """Export model to ONNX format."""
        try:
            import tf2onnx
            import onnx
        except ImportError:
            print("tf2onnx/onnx not installed. Skipping ONNX export.")
            return None

        output_path = self.output_dir / filename
        if input_signature is None:
            input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]

        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        onnx.save(onnx_model, str(output_path))
        print(f"Model exported to ONNX: {output_path}")
        return output_path

    def export_tflite(
        self,
        model: tf.keras.Model,
        filename: str = "model_best.tflite",
        optimize: bool = True,
    ) -> Path:
        """Export model to TensorFlow Lite format."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        output_path = self.output_dir / filename
        output_path.write_bytes(tflite_model)
        print(f"Model exported to TFLite: {output_path}")
        return output_path

    def export_all(self, model: tf.keras.Model, prefix: str = "classifier") -> dict:
        """Export model to all supported formats."""
        results = {}
        results["keras"] = self.export_keras(model, f"{prefix}_best.keras")
        results["h5"] = self.export_h5(model, f"{prefix}_best.h5")
        results["onnx"] = self.export_onnx(model, f"{prefix}_best.onnx")
        results["tflite"] = self.export_tflite(model, f"{prefix}_best.tflite")
        return results
