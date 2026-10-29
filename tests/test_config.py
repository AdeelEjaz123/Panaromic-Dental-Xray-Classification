"""Test configuration and clinical taxonomies."""

from dental_vision.config import (
    CLINICAL_CATEGORIES,
    COLOR_PALETTE,
    DENTAL_CLASSES,
    DEFAULT_CONFIG,
    get_class_color,
)


def test_dental_classes_not_empty():
    assert len(DENTAL_CLASSES) > 0
    assert "Caries" in DENTAL_CLASSES
    assert "Crown" in DENTAL_CLASSES
    assert "Filling" in DENTAL_CLASSES
    assert "Implant" in DENTAL_CLASSES


def test_clinical_categories_coverage():
    assert "Pathology & Lesions" in CLINICAL_CATEGORIES
    assert "Restorative & Prosthodontics" in CLINICAL_CATEGORIES
    assert "Endodontics & Surgery" in CLINICAL_CATEGORIES
    assert "Orthodontics & Alignment" in CLINICAL_CATEGORIES


def test_get_class_color():
    color = get_class_color("Caries")
    assert isinstance(color, tuple)
    assert len(color) == 3

    # Fallback check
    fallback = get_class_color("UnknownCondition")
    assert isinstance(fallback, tuple)
    assert len(fallback) == 3


def test_default_config_keys():
    required_keys = ["imgsz", "conf", "iou", "batch", "epochs"]
    for k in required_keys:
        assert k in DEFAULT_CONFIG
