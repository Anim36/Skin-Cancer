"""Dataset loading and metadata merging for HAM10000, ISIC 2019, and ISIC 2020."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocessing.augmentation import AugmentationPipeline
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import Config


@dataclass
class SampleRecord:
    """Single dataset sample with image path and metadata."""

    image_path: Path
    label: int
    label_name: str
    age: float
    gender: int
    location: int
    image_id: str


class DatasetManager:
    """Load, merge, and prepare skin lesion datasets."""

    DATASET_SCHEMAS = {
        "ham10000": {
            "metadata_patterns": ["HAM10000_metadata.csv", "metadata.csv"],
            "image_col": "image_id",
            "label_col": "dx",
            "age_col": "age",
            "gender_col": "sex",
            "location_col": "localization",
            "image_dirs": ["images", "HAM10000_images", ""],
        },
        "isic2019": {
            "metadata_patterns": ["ISIC_2019_Training_GroundTruth.csv", "metadata.csv"],
            "image_col": "image",
            "label_col": "label",
            "age_col": "age_approx",
            "gender_col": "sex",
            "location_col": "anatom_site_general",
            "image_dirs": ["ISIC_2019_Training_Input", "images", ""],
        },
        "isic2020": {
            "metadata_patterns": ["ISIC_2020_Training_GroundTruth.csv", "train.csv", "metadata.csv"],
            "image_col": "image_name",
            "label_col": "benign_malignant",
            "age_col": "age_approx",
            "gender_col": "sex",
            "location_col": "anatom_site_general_challenge",
            "image_dirs": ["ISIC_2020_Training_JPEG", "train", "images", ""],
        },
    }

    ISIC2019_LABELS = [
        "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK",
    ]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_name = config.get("data.dataset_name", "ham10000").lower()
        self.class_names = config.class_names
        self.preprocessor = PreprocessingPipeline(config)
        self.augmentation = AugmentationPipeline(config)
        self.label_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.samples: List[SampleRecord] = []
        self.metadata_df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Discover and load dataset with merged metadata."""
        schema = self.DATASET_SCHEMAS.get(self.dataset_name)
        if schema is None:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        dataset_dir = self._find_dataset_dir()
        metadata_path = self._find_metadata(dataset_dir, schema["metadata_patterns"])
        images_dir = self._find_images_dir(dataset_dir, schema["image_dirs"])

        self.metadata_df = self._read_metadata(metadata_path, schema)
        self.samples = self._build_samples(self.metadata_df, images_dir, schema)
        print(f"Loaded {len(self.samples)} samples from {self.dataset_name}")
        return self.metadata_df

    def _find_dataset_dir(self) -> Path:
        candidates = [
            self.config.dataset_root / self.dataset_name,
            self.config.dataset_root,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Dataset directory not found for {self.dataset_name}")

    @staticmethod
    def _find_metadata(dataset_dir: Path, patterns: List[str]) -> Path:
        for pattern in patterns:
            path = dataset_dir / "metadata" / pattern
            if path.exists():
                return path
            path = dataset_dir / pattern
            if path.exists():
                return path
        csv_files = list(dataset_dir.rglob("*.csv"))
        if csv_files:
            return csv_files[0]
        raise FileNotFoundError(f"No metadata CSV found in {dataset_dir}")

    @staticmethod
    def _find_images_dir(dataset_dir: Path, image_dirs: List[str]) -> Path:
        for name in image_dirs:
            path = dataset_dir / name if name else dataset_dir
            if path.exists() and (any(path.glob("*.jpg")) or any(path.glob("*.png"))):
                return path
        for pattern in ("*.jpg", "*.png", "*.jpeg"):
            if list(dataset_dir.rglob(pattern)):
                return dataset_dir
        raise FileNotFoundError(f"No image directory found in {dataset_dir}")

    def _read_metadata(self, metadata_path: Path, schema: Dict) -> pd.DataFrame:
        df = pd.read_csv(metadata_path)
        if self.dataset_name == "isic2019" and schema["label_col"] not in df.columns:
            label_cols = [c for c in self.ISIC2019_LABELS if c in df.columns]
            df["label"] = df[label_cols].idxmax(axis=1)
        return df

    def _build_samples(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        schema: Dict,
    ) -> List[SampleRecord]:
        gender_classes = self.config.get("metadata.gender_classes", ["male", "female", "unknown"])
        location_classes = self.config.get("metadata.location_classes", [])

        labels = df[schema["label_col"]].astype(str).str.lower().tolist()
        self.label_encoder.fit(self.class_names)
        mapped_labels = []
        for label in labels:
            if label in self.class_names:
                mapped_labels.append(label)
            elif label == "mel":
                mapped_labels.append("mel")
            elif label in ("benign", "malignant"):
                mapped_labels.append("mel" if label == "malignant" else "nv")
            else:
                mapped_labels.append(self.class_names[0])

        genders = df.get(schema["gender_col"], pd.Series(["unknown"] * len(df)))
        genders = genders.fillna("unknown").astype(str).str.lower()
        self.gender_encoder.fit(gender_classes)
        gender_encoded = self.gender_encoder.transform(
            [g if g in gender_classes else "unknown" for g in genders]
        )

        locations = df.get(schema["location_col"], pd.Series(["unknown"] * len(df)))
        locations = locations.fillna("unknown").astype(str).str.lower()
        self.location_encoder.fit(location_classes)
        location_encoded = self.location_encoder.transform(
            [loc if loc in location_classes else "unknown" for loc in locations]
        )

        ages = df.get(schema["age_col"], pd.Series([50.0] * len(df)))
        ages = ages.fillna(50.0).astype(float)
        age_min = self.config.get("metadata.age_min", 0)
        age_max = self.config.get("metadata.age_max", 100)
        ages = np.clip(ages, age_min, age_max) / age_max

        samples: List[SampleRecord] = []
        image_ids = df[schema["image_col"]].astype(str).tolist()
        label_indices = self.label_encoder.transform(mapped_labels)

        for idx, image_id in enumerate(image_ids):
            image_path = self._resolve_image_path(images_dir, image_id)
            if image_path is None:
                continue
            samples.append(
                SampleRecord(
                    image_path=image_path,
                    label=int(label_indices[idx]),
                    label_name=mapped_labels[idx],
                    age=float(ages.iloc[idx]),
                    gender=int(gender_encoded[idx]),
                    location=int(location_encoded[idx]),
                    image_id=image_id,
                )
            )
        return samples

    @staticmethod
    def _resolve_image_path(images_dir: Path, image_id: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
            candidate = images_dir / image_id
            if candidate.exists():
                return candidate
        matches = list(images_dir.glob(f"{image_id}*"))
        return matches[0] if matches else None

    def split_data(
        self,
        validation_split: Optional[float] = None,
        test_split: Optional[float] = None,
    ) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
        """Split samples into train, validation, and test sets."""
        val_split = validation_split or self.config.get("data.validation_split", 0.15)
        test_split_val = test_split or self.config.get("data.test_split", 0.15)
        labels = [s.label for s in self.samples]

        train_samples, temp_samples = train_test_split(
            self.samples,
            test_size=val_split + test_split_val,
            stratify=labels,
            random_state=self.config.get("project.seed", 42),
        )
        relative_test = test_split_val / (val_split + test_split_val)
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=relative_test,
            stratify=[s.label for s in temp_samples],
            random_state=self.config.get("project.seed", 42),
        )
        return train_samples, val_samples, test_samples

    def _load_image_numpy(self, path: str, image_size: Tuple[int, int]) -> np.ndarray:
        """Load and preprocess a single image."""
        import cv2

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.preprocessor.process(image, target_size=image_size)

    def _generator(
        self,
        samples: List[SampleRecord],
        image_size: Tuple[int, int],
        augment: bool = False,
        for_segmentation: bool = False,
    ):
        """Python generator for tf.data.Dataset."""
        for sample in samples:
            image = self._load_image_numpy(str(sample.image_path), image_size)
            if augment and self.augmentation.enabled:
                image = self.augmentation.augment(image)

            if for_segmentation:
                mask = self._pseudo_mask_numpy(image)
                yield image.astype(np.float32), mask.astype(np.float32)
            else:
                metadata = np.array([sample.age, float(sample.gender), float(sample.location)], dtype=np.float32)
                label = np.zeros(self.config.num_classes, dtype=np.float32)
                label[sample.label] = 1.0
                yield (image.astype(np.float32), metadata), label

    def create_tf_dataset(
        self,
        samples: List[SampleRecord],
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        augment: bool = False,
        for_segmentation: bool = False,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from sample records."""
        batch_size = batch_size or self.config.get("data.batch_size", 16)
        image_size = (
            tuple(self.config.get("segmentation.input_size", [256, 256]))
            if for_segmentation
            else self.config.image_size
        )

        output_signature = (
            (
                tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(self.config.num_classes,), dtype=tf.float32),
        ) if not for_segmentation else (
            tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*image_size, 1), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: self._generator(samples, image_size, augment, for_segmentation),
            output_signature=output_signature,
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(samples), 1000))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def _pseudo_mask_numpy(image: np.ndarray) -> np.ndarray:
        """Generate pseudo segmentation mask using intensity thresholding."""
        gray = np.mean(image, axis=-1, keepdims=True)
        threshold = gray.mean() - 0.05
        return (gray < threshold).astype(np.float32)

    def get_metadata_vocab_sizes(self) -> Dict[str, int]:
        """Return vocabulary sizes for categorical metadata."""
        return {
            "gender": len(self.config.get("metadata.gender_classes", [])),
            "location": len(self.config.get("metadata.location_classes", [])),
        }

    def get_class_weights(self) -> Dict[int, float]:
        """Compute balanced class weights."""
        labels = [s.label for s in self.samples]
        counts = np.bincount(labels, minlength=self.config.num_classes)
        total = len(labels)
        weights = {
            i: total / (self.config.num_classes * count) if count > 0 else 1.0
            for i, count in enumerate(counts)
        }
        return weights
