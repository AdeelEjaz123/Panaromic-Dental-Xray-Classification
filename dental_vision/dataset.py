"""Dataset utilities for Dental Vision: YAML generation, validation, and subset creation."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

from dental_vision.config import DENTAL_CLASSES


class DentalDataset:
    """Helper to inspect, validate, and generate dataset YAML configurations."""

    @staticmethod
    def create_yaml(
        dataset_dir: Union[str, Path],
        output_yaml_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        train_rel: str = "train/images",
        val_rel: str = "valid/images",
        test_rel: Optional[str] = "test/images",
    ) -> str:
        """Generate a valid YOLO data.yaml configuration file.

        Args:
            dataset_dir: Root path of dataset containing train/valid/test folders.
            output_yaml_path: Destination path for the .yaml file.
            class_names: List of class names. Defaults to DENTAL_CLASSES.
            train_rel: Subpath to training images.
            val_rel: Subpath to validation images.
            test_rel: Subpath to test images (optional).

        Returns:
            Absolute path to generated YAML file.
        """
        dataset_dir = Path(dataset_dir).resolve()
        class_names = class_names or DENTAL_CLASSES

        config: Dict[str, Union[str, int, Dict[int, str]]] = {
            "path": str(dataset_dir),
            "train": train_rel,
            "val": val_rel,
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }
        if test_rel:
            config["test"] = test_rel

        output_path = Path(output_yaml_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"[Dataset]  Generated dataset configuration YAML at: {output_path}")
        return str(output_path)

    @staticmethod
    def validate_structure(dataset_dir: Union[str, Path]) -> Dict[str, int]:
        """Validate standard YOLO directory structure and return image counts."""
        dataset_path = Path(dataset_dir).resolve()
        stats = {}
        for split in ["train", "valid", "val", "test"]:
            img_dir = dataset_path / split / "images"
            if not img_dir.exists():
                img_dir = dataset_path / split
            if img_dir.exists() and img_dir.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                images = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts]
                stats[split] = len(images)
        return stats
