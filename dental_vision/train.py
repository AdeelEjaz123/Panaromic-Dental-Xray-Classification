"""Production Training Pipeline for Dental Vision."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from ultralytics import YOLO

from dental_vision.config import DEFAULT_CONFIG


class DentalTrainer:
    """Trainer orchestrating YOLO model fine-tuning on panoramic dental X-rays."""

    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        device: Optional[Union[int, str]] = None,
    ):
        self.base_model = base_model
        if device is None or device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[Init]  Initializing DentalTrainer on device: {self.device}")

    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        patience: int = 15,
        dropout: float = 0.1,
        weight_decay: float = 0.0005,
        lr0: float = 0.001,
        project: str = "runs/train",
        name: str = "dental_run",
        fraction: Optional[float] = None,
        cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run training with medical imaging augmentations & regularization.

        Args:
            data_yaml: Path to dataset YAML file.
            epochs: Max training epochs (early stopping handles convergence).
            batch: Batch size (default: 16).
            imgsz: Resolution (default: 640).
            patience: Early stopping patience epochs.
            dropout: Node dropout rate for regularization.
            weight_decay: L2 penalty.
            lr0: Initial learning rate.
            project: Output directory root.
            name: Experiment name.
            fraction: Dataset fraction to use (e.g. 0.1 for 10% subset test).
            cache: Cache images into RAM.
            **kwargs: Extra arguments passed directly to YOLO.train.

        Returns:
            Ultralytics training results object.
        """
        data_path = str(Path(data_yaml).resolve())
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset YAML not found at: {data_path}")

        print(f"[Model]  Loading pre-trained base model: {self.base_model}")
        model = YOLO(self.base_model)

        train_args: Dict[str, Any] = {
            "data": data_path,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": self.device,
            "patience": patience,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "lr0": lr0,
            "optimizer": "auto",
            "project": project,
            "name": name,
            "plots": True,
            "cache": cache,
            # Medical imaging augmentations
            "mosaic": 1.0,
            "mixup": 0.05,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        }

        if fraction is not None:
            train_args["fraction"] = fraction

        # Update with any user-overridden kwargs
        train_args.update(kwargs)

        print(f"[Training]  Starting training: {epochs} epochs | batch={batch} | imgsz={imgsz}")
        results = model.train(**train_args)
        print(f"[OK]  Training completed! Model saved to {project}/{name}/weights/best.pt")
        return results


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO model on Panoramic Dental X-ray dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (e.g., configs/dental_pathology.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model checkpoint (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('auto', 'cpu', '0', etc.)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dental_run",
        help="Experiment name",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Dataset subset fraction (e.g. 0.1 for 10% test run)",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    trainer = DentalTrainer(base_model=args.model, device=args.device)
    trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        fraction=args.fraction,
    )


if __name__ == "__main__":
    main()
