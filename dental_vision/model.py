"""DentalModel: Wrapper around Ultralytics YOLO with medical imaging utilities."""

import os
from typing import Optional, Union
import torch
from ultralytics import YOLO


class DentalModel:
    """Wrapper around YOLOv8 model for dental panoramic X-ray classification and detection."""

    def __init__(self, model_path_or_name: str = "weights/best.pt", device: Optional[str] = None):
        """Initialize DentalModel.

        Args:
            model_path_or_name: Path to trained .pt weights (defaults to 'weights/best.pt' or 'run_1-2/weights/best.pt').
            device: 'cpu', 'cuda', '0', or None for automatic hardware detection.
        """
        # Resolve weights path if default
        resolved_path = model_path_or_name
        if not os.path.exists(resolved_path):
            if os.path.exists("weights/best.pt"):
                resolved_path = "weights/best.pt"
            elif os.path.exists("run_1-2/weights/best.pt"):
                resolved_path = "run_1-2/weights/best.pt"

        self.model_path = resolved_path
        self.device = self._resolve_device(device)
        self.model = YOLO(self.model_path)

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> Union[str, int]:
        """Automatically resolve target computation device."""
        if device is not None and device.lower() != "auto":
            return device
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def names(self) -> dict:
        """Return class index-to-name mapping from the underlying model."""
        return self.model.names if hasattr(self.model, "names") else {}

    def export(self, format: str = "onnx", imgsz: int = 640, dynamic: bool = False) -> str:
        """Export model to production inference formats (ONNX, TorchScript, TensorRT, CoreML).

        Args:
            format: Target format ('onnx', 'torchscript', 'engine', 'tflite').
            imgsz: Input resolution.
            dynamic: Enable dynamic axes for varying batch sizes and resolutions.

        Returns:
            Path to exported artifact.
        """
        print(f"[Model]  Exporting Dental YOLO model to format: {format} (imgsz={imgsz})...")
        export_path = self.model.export(format=format, imgsz=imgsz, dynamic=dynamic)
        print(f"[OK]  Export completed successfully: {export_path}")
        return export_path

    def __repr__(self) -> str:
        return f"DentalModel(model='{self.model_path}', device='{self.device}', classes={len(self.names)})"
