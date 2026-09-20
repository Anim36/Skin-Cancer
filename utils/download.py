"""Automatic dataset download utilities."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from utils.config import Config


class DatasetDownloader:
    """Download supported skin lesion datasets."""

    DATASET_URLS = {
        "ham10000": {
            "kaggle": "kmader/skin-cancer-mnist-ham10000",
            "description": "HAM10000 dermatoscopic images",
        },
        "isic2019": {
            "url": "https://challenge.isic-archive.com/data/2019/",
            "description": "ISIC 2019 Challenge",
        },
        "isic2020": {
            "url": "https://challenge.isic-archive.com/data/2020/",
            "description": "ISIC 2020 Challenge",
        },
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_root = config.dataset_root
        self.dataset_root.mkdir(parents=True, exist_ok=True)

    def download(self, dataset_name: Optional[str] = None) -> Path:
        """
        Download dataset based on configuration.

        Args:
            dataset_name: Override dataset name from config.

        Returns:
            Path to downloaded dataset directory.
        """
        name = (dataset_name or self.config.get("data.dataset_name", "ham10000")).lower()
        if name == "ham10000":
            return self._download_ham10000()
        if name in ("isic2019", "isic2020"):
            return self._download_isic_instructions(name)
        raise ValueError(f"Unsupported dataset for auto-download: {name}")

    def _download_ham10000(self) -> Path:
        """Download HAM10000 from Kaggle if credentials are available."""
        target_dir = self.dataset_root / "ham10000"
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"HAM10000 already exists at {target_dir}")
            return target_dir

        kaggle_dataset = self.config.get("download.kaggle_dataset", "kmader/skin-cancer-mnist-ham10000")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(target_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            for zip_file in target_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_file, "r") as archive:
                    archive.extractall(target_dir)
                zip_file.unlink()
            print(f"HAM10000 downloaded to {target_dir}")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(
                "Kaggle CLI unavailable or download failed. "
                f"Place HAM10000 files manually in {target_dir}. Error: {exc}"
            )
        return target_dir

    def _download_isic_instructions(self, dataset_name: str) -> Path:
        """Provide ISIC download guidance; manual download required."""
        target_dir = self.dataset_root / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        info = self.DATASET_URLS[dataset_name]
        readme_path = target_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        readme_path.write_text(
            f"Please download {info['description']} from {info['url']}\n"
            f"Extract images and metadata CSV into: {target_dir}\n",
            encoding="utf-8",
        )
        print(f"ISIC datasets require manual download. Instructions saved to {readme_path}")
        return target_dir

    @staticmethod
    def organize_ham10000(source_dir: Path, target_dir: Path) -> None:
        """Organize extracted HAM10000 files into images/ and metadata/ folders."""
        images_dir = target_dir / "images"
        metadata_dir = target_dir / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for csv_file in source_dir.glob("*.csv"):
            shutil.copy2(csv_file, metadata_dir / csv_file.name)

        for pattern in ("*.jpg", "*.png", "*.jpeg"):
            for image_path in source_dir.rglob(pattern):
                if "images" not in image_path.parts:
                    dest = images_dir / image_path.name
                    if not dest.exists():
                        shutil.copy2(image_path, dest)
