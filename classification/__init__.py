"""Classification module with EfficientNetV2 and metadata fusion."""

from classification.efficientnet_model import HybridClassifier
from classification.metadata_fusion import MetadataFusion

__all__ = ["HybridClassifier", "MetadataFusion"]
