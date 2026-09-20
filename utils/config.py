"""Configuration management for the Skin Cancer Detection System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class Config:
    """Central configuration container loaded from YAML."""

    raw: Dict[str, Any] = field(default_factory=dict)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve nested configuration value using dot notation."""
        keys = key.split(".")
        value: Any = self.raw
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def set(self, key: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = key.split(".")
        target = self.raw
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.get("project.output_dir", "outputs")

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / self.get("project.checkpoint_dir", "checkpoints")

    @property
    def log_dir(self) -> Path:
        return self.project_root / self.get("project.log_dir", "logs")

    @property
    def dataset_root(self) -> Path:
        return self.project_root / self.get("data.dataset_root", "dataset")

    @property
    def metadata_dir(self) -> Path:
        return self.project_root / self.get("data.metadata_dir", "metadata")

    @property
    def image_size(self) -> tuple[int, int]:
        size = self.get("data.image_size", [224, 224])
        return int(size[0]), int(size[1])

    @property
    def num_classes(self) -> int:
        return int(self.get("data.num_classes", 7))

    @property
    def class_names(self) -> List[str]:
        return list(self.get("data.class_names", []))

    def ensure_directories(self) -> None:
        """Create required output directories."""
        for directory in [
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.dataset_root,
            self.metadata_dir,
            self.output_dir / "gradcam",
            self.output_dir / "plots",
            self.output_dir / "reports",
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config. Defaults to configs/default.yaml.
        overrides: Optional dictionary of values to merge.

    Returns:
        Config instance.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if overrides:
        raw_config = _deep_merge(raw_config, overrides)

    config = Config(raw=raw_config)
    config.ensure_directories()
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
