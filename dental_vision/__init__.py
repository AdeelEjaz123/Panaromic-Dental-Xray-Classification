"""Dental Vision: Deep Learning Suite for Panoramic Dental X-ray Tooth Localization & Pathology Detection."""

__version__ = "0.1.0"
__author__ = "Dental Vision Contributors"

from dental_vision.config import (
    CLINICAL_CATEGORIES,
    COLOR_PALETTE,
    DENTAL_CLASSES,
    DEFAULT_CONFIG,
    get_class_color,
)

__all__ = [
    "DENTAL_CLASSES",
    "COLOR_PALETTE",
    "CLINICAL_CATEGORIES",
    "DEFAULT_CONFIG",
    "get_class_color",
    "DentalModel",
    "DentalPredictor",
    "DentalTrainer",
    "DentalVisualizer",
    "DentalDataset",
]


def __getattr__(name: str):
    if name == "DentalModel":
        from dental_vision.model import DentalModel
        return DentalModel
    if name == "DentalPredictor":
        from dental_vision.predict import DentalPredictor
        return DentalPredictor
    if name == "DentalTrainer":
        from dental_vision.train import DentalTrainer
        return DentalTrainer
    if name == "DentalVisualizer":
        from dental_vision.visualize import DentalVisualizer
        return DentalVisualizer
    if name == "DentalDataset":
        from dental_vision.dataset import DentalDataset
        return DentalDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
