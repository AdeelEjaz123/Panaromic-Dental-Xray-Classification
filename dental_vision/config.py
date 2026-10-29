"""Configuration, clinical taxonomies, and default hyperparameters for Dental Vision."""

from typing import Dict, List, Tuple

# Comprehensive multi-class dental pathology and anatomic structure taxonomy
DENTAL_CLASSES: List[str] = [
    "Tooth",
    "Caries",
    "Crown",
    "Filling",
    "Implant",
    "Malaligned",
    "Mandibular Canal",
    "Missing teeth",
    "Periapical lesion",
    "Retained root",
    "Root Canal Treatment",
    "Root Piece",
    "impacted tooth",
    "Maxillary sinus",
    "Bone Loss",
    "Supra Eruption",
    "abutment",
    "metal band",
    "orthodontic brackets",
    "post - core",
    "wire",
]

# Clinical categories grouping related dental findings
CLINICAL_CATEGORIES: Dict[str, List[str]] = {
    "Pathology & Lesions": [
        "Caries",
        "Periapical lesion",
        "Bone Loss",
    ],
    "Restorative & Prosthodontics": [
        "Crown",
        "Filling",
        "Implant",
        "abutment",
        "post - core",
    ],
    "Endodontics & Surgery": [
        "Root Canal Treatment",
        "Retained root",
        "Root Piece",
    ],
    "Orthodontics & Alignment": [
        "Malaligned",
        "Supra Eruption",
        "metal band",
        "orthodontic brackets",
        "wire",
    ],
    "Anatomy & Landmarks": [
        "Tooth",
        "Missing teeth",
        "impacted tooth",
        "Mandibular Canal",
        "Maxillary sinus",
    ],
}

# Hex/RGB Colors organized by clinical category for intuitive X-ray overlay
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Pathology & Lesions": (230, 57, 70),       # High-alert Red
    "Restorative & Prosthodontics": (69, 123, 157), # Clinical Blue
    "Endodontics & Surgery": (233, 196, 106),   # Amber Yellow
    "Orthodontics & Alignment": (42, 157, 143),  # Cyan / Teal
    "Anatomy & Development": (108, 117, 125),   # Cool Slate Gray
}

# Distinct BGR color palette for individual class labels
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Caries": (0, 0, 255),                  # Red
    "Periapical lesion": (0, 69, 255),      # Orange-Red
    "Bone Loss": (0, 140, 255),             # Dark Orange
    "Crown": (255, 191, 0),                 # Deep Sky Blue
    "Filling": (255, 144, 30),              # Dodger Blue
    "Implant": (238, 130, 238),             # Violet
    "abutment": (205, 92, 92),              # Indian Red
    "post - core": (147, 112, 219),         # Medium Purple
    "Root Canal Treatment": (0, 215, 255),  # Gold
    "Retained root": (34, 139, 34),         # Forest Green
    "Root Piece": (50, 205, 50),            # Lime Green
    "Malaligned": (255, 165, 0),            # Orange
    "Supra Eruption": (255, 20, 147),       # Deep Pink
    "metal band": (192, 192, 192),          # Silver
    "orthodontic brackets": (0, 255, 127),  # Spring Green
    "wire": (72, 209, 204),                 # Medium Turquoise
    "Missing teeth": (128, 128, 128),       # Gray
    "impacted tooth": (255, 0, 255),        # Magenta
    "Tooth": (0, 255, 0),                   # Green
}

def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Return BGR color tuple for a given class name, with fallback."""
    if class_name in COLOR_PALETTE:
        return COLOR_PALETTE[class_name]
    # Check clinical category
    for cat, classes in CLINICAL_CATEGORIES.items():
        if class_name.lower() in [c.lower() for c in classes]:
            rgb = CATEGORY_COLORS[cat]
            return (rgb[2], rgb[1], rgb[0])
    return (0, 255, 0)

# Default training & inference parameters
DEFAULT_CONFIG = {
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.45,
    "batch": 16,
    "epochs": 50,
    "patience": 15,
    "dropout": 0.1,
    "weight_decay": 0.0005,
    "lr0": 0.001,
    "base_model": "yolov8n.pt",
    "default_weights": "weights/best.pt",
}
