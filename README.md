# 🦷 Dental Vision AI: Panoramic Dental Radiograph Pathology & Tooth Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Model: YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20Nano-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An end-to-end deep learning computer vision pipeline engineered for automated multi-condition pathology triage, individual tooth localization, and diagnostic reporting on Panoramic Dental Radiographs (Orthopantomograms - OPG).

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Clinical Taxonomy & Supported Classes](#-clinical-taxonomy--supported-classes)
- [Model Architecture & Benchmark](#-model-architecture--benchmark)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [1. Inference via CLI](#1-inference-via-cli)
  - [2. Programmatic Python API](#2-programmatic-python-api)
  - [3. Clinical Triage JSON Report](#3-clinical-triage-json-report)
  - [4. Training & Fine-Tuning](#4-training--fine-tuning)
  - [5. Model Export (ONNX / TorchScript)](#5-model-export-onnx--torchscript)
- [Interactive Notebook](#-interactive-notebook)
- [Clinical & Ethical Disclaimer](#-clinical--ethical-disclaimer)
- [License & Contributing](#-license--contributing)

---

## 🌟 Executive Overview

Dental panoramic radiographs (OPGs) are foundational for clinical dentistry, yet manual evaluation of full-arch scans is prone to cognitive fatigue and inter-observer variability. **Dental Vision AI** addresses this challenge by providing:
- **Simultaneous Multi-Pathology Detection:** Rapidly scans full OPGs to detect caries, periapical lesions, bone loss, crowns, fillings, implants, and root canal treatments.
- **Single-Tooth Isolation:** Automatically isolates individual teeth into multi-panel contact sheets for focused tooth-by-tooth diagnosis.
- **Automated Clinical Reports:** Generates structured JSON summaries categorized by dental discipline (Restorative, Endodontic, Orthodontic, Pathology).
- **Sub-10ms Inference:** Lightweight edge-deployable YOLOv8 Nano architecture operating at over 150 FPS on modern GPUs.

---

## 🚀 Key Features

- **Transfer Learning on YOLOv8:** Leverages anchor-free object detection with decoupled heads for bounding box regression and classification.
- **Medical Radiography Augmentation:** Tailored data pipeline using mosaic (1.0), mixup (0.05), anatomical horizontal flipping, and multi-scale scaling.
- **Structured Clinical Triage:** Maps raw bounding boxes into high-level categories (Pathology, Restorative, Endodontics, Orthodontics, Anatomy).
- **Production-Ready Package:** Full Python library (`pip install -e .`), unified CLI (`main.py`), type hints, and automated CI tests.
- **Export Ready:** Seamless export to ONNX, TorchScript, and TensorRT for deployment on clinical workstations and PACS systems.

---

## 🩺 Clinical Taxonomy & Supported Classes

The system detects **19+ dental conditions and structures**, organized into 5 clinical specialties:

| Clinical Discipline | Detected Findings / Structures | Clinical Purpose |
| :--- | :--- | :--- |
| **Pathology & Lesions** | `Caries`, `Periapical lesion`, `Bone Loss` | Early detection of decay, apical periodontitis, and periodontal disease. |
| **Restorative & Prosthodontics** | `Crown`, `Filling`, `Implant`, `abutment`, `post - core` | Assessment of existing restorations and implant stability. |
| **Endodontics & Surgery** | `Root Canal Treatment`, `Retained root`, `Root Piece` | Verification of obturation quality and retained root fragments. |
| **Orthodontics & Alignment** | `Malaligned`, `Supra Eruption`, `metal band`, `orthodontic brackets`, `wire` | Malocclusion tracking and orthodontic appliance monitoring. |
| **Anatomy & Development** | `Tooth`, `Missing teeth`, `impacted tooth` | Full arch tooth numbering, impaction screening (e.g., 3rd molars). |

---

## 📊 Model Architecture & Benchmark

```mermaid
graph LR
    A["Raw Panoramic OPG (640x640)"] --> B["YOLOv8n Backbone (Conv + C2f)"]
    B --> C["Feature Pyramid (PAN-FPN)"]
    C --> D["Decoupled Detection Heads"]
    D --> E["Non-Maximum Suppression (IoU: 0.45)"]
    E --> F["Pathology Localization & Classification"]
    E --> G["Clinical Triage JSON Summary"]
```

### Benchmark Performance (YOLOv8 Nano on Test Set)

| Metric | Value | Clinical Significance |
| :--- | :--- | :--- |
| **Architecture** | YOLOv8n (Anchor-Free) | 73 layers, 3.14M parameters |
| **Computational Footprint** | 8.7 GFLOPs | Ultra-lightweight edge footprint |
| **Inference Latency** | **4.9 ms / image** | Benchmarked on NVIDIA Tesla T4 GPU (~150+ FPS) |
| **mAP@50 (Implant)** | **0.965** | Near-perfect sensitivity on metallic hardware |
| **mAP@50 (Impacted Tooth)** | **0.907** | High accuracy on 3rd molar impactions |
| **mAP@50 (Crown)** | **0.795** | Reliable crown boundary localization |
| **mAP@50 (Missing Teeth)** | **0.694** | Accurate edentulous space detection |

---

## 📈 Empirical Training Results & Visualizations

The official 50-epoch benchmark run artifacts are preserved in [`run_1-2/`](run_1-2/) and the production weights are provided in [`weights/best.pt`](weights/best.pt).

### 1. Training & Validation Metric Trajectories
Loss convergence (Box, Classification, DFL) and progressive mAP@50 improvements across 50 epochs:

<p align="center">
  <img src="run_1-2/results.png" alt="Training and Validation Metrics" width="90%"/>
</p>

### 2. Multi-Class Normalized Confusion Matrix & Precision-Recall
Detailed class-level diagnostic confusion and PR curves across confidence thresholds:

<p align="center">
  <img src="run_1-2/confusion_matrix_normalized.png" alt="Normalized Confusion Matrix" width="48%"/>
  <img src="run_1-2/BoxPR_curve.png" alt="Precision-Recall Curve" width="48%"/>
</p>

### 3. Model Predictions on Validation Panoramic Radiographs
Real OPG radiograph validation detections demonstrating multi-condition pathology bounding boxes:

<p align="center">
  <img src="run_1-2/val_batch0_pred.jpg" alt="Validation Batch Predictions" width="85%"/>
</p>

---

## 📂 Project Architecture

```
dental-vision/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions test pipeline
├── configs/
│   ├── dental_pathology.yaml    # Dataset split definition & class mappings
│   └── hyperparameters.yaml     # Fine-tuning parameters & augmentations
├── dental_vision/               # Core Python package
│   ├── __init__.py              # Package exports & lazy imports
│   ├── config.py                # Class taxonomies, palettes & constants
│   ├── dataset.py               # Dataset validation & YAML generator
│   ├── model.py                 # YOLO wrapper & model export engine
│   ├── train.py                 # Training pipeline with early stopping
│   ├── predict.py               # Clinical inference & JSON reporter
│   └── visualize.py             # Tooth isolation grids & side-by-side plots
├── notebooks/
│   └── 01_dental_xray_training_and_inference.ipynb  # End-to-end interactive workflow
├── run_1-2/                     # Complete benchmark training run output
│   ├── weights/                 # Checkpoints (best.pt & last.pt)
│   ├── results.png              # 50-epoch loss & mAP curves
│   ├── confusion_matrix.png     # Multi-class confusion matrices
│   ├── BoxPR_curve.png          # Precision-Recall curves
│   └── val_batch*_pred.jpg      # Validation sample predictions
├── weights/
│   └── best.pt                  # Trained production weights (6.2 MB)
├── tests/
│   ├── __init__.py
│   ├── test_config.py           # Taxonomy & configuration tests
│   └── test_dataset.py          # Dataset YAML generator tests
├── main.py                      # Unified CLI entrypoint
├── pyproject.toml               # Modern Python packaging configuration
├── setup.py                     # Setuptools packaging script
├── requirements.txt             # Pinned project dependencies
├── LICENSE                      # MIT Open Source License
├── CONTRIBUTING.md              # Guidelines for open-source contributors
└── README.md                    # Comprehensive documentation
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/dental-xray-classification.git
cd dental-xray-classification
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Quick Start

### 1. Inference via CLI
Run multi-pathology detection on a panoramic X-ray (automatically uses trained `weights/best.pt`):
```bash
python main.py predict --source samples/opg_sample.jpg --conf 0.25 --output output/annotated.jpg --report output/report.json
```

Or process an entire directory of X-rays:
```bash
python main.py predict --source /path/to/xrays/ --output output/batch_results/
```

### 2. Programmatic Python API
```python
from dental_vision import DentalPredictor, DentalVisualizer

# 1. Initialize predictor
predictor = DentalPredictor(weights_path="weights/best.pt", conf=0.25)

# 2. Run inference
findings = predictor.predict("samples/opg_sample.jpg")
print(f"Total dental conditions identified: {len(findings)}")

for item in findings:
    print(f"[{item['clinical_category']}] {item['class_name']} - Confidence: {item['confidence']:.2f}")

# 3. Generate side-by-side diagnostic panel
DentalVisualizer.side_by_side_comparison(
    raw_image_path="samples/opg_sample.jpg",
    annotated_image_path="output/annotated.jpg",
    output_path="output/diagnostic_panel.jpg"
)
```

### 3. Clinical Triage JSON Report
The inference engine automatically produces structured diagnostic reports for electronic dental records (EDR):

```json
{
  "source_image": "opg_sample.jpg",
  "total_findings": 5,
  "confidence_threshold": 0.25,
  "category_summary": {
    "Pathology & Lesions": 2,
    "Restorative & Prosthodontics": 2,
    "Endodontics & Surgery": 1
  },
  "class_summary": {
    "Caries": 1,
    "Periapical lesion": 1,
    "Crown": 1,
    "Filling": 1,
    "Root Canal Treatment": 1
  },
  "findings": [
    {
      "class_id": 1,
      "class_name": "Caries",
      "confidence": 0.88,
      "box_xyxy": [182.4, 290.1, 235.8, 350.6],
      "clinical_category": "Pathology & Lesions"
    }
  ]
}
```

### 4. Training & Fine-Tuning
Train the model on your custom annotated OPG dataset:
```bash
python main.py train \
  --data configs/dental_pathology.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device auto \
  --name dental_pathology_run
```

### 5. Model Export (ONNX / TorchScript)
Export trained weights for edge deployment:
```bash
python main.py export --weights weights/best.pt --format onnx --imgsz 640
```

---

## 📓 Interactive Notebook

An interactive, self-contained Jupyter notebook is provided in [`notebooks/01_dental_xray_training_and_inference.ipynb`](file:///notebooks/01_dental_xray_training_and_inference.ipynb). It guides you through:
1. Environment setup & GPU verification
2. Dataset loading and augmentation pipelines
3. Transfer learning execution with early stopping
4. Visual inspection and single-tooth isolation grids

---

## ⚕️ Clinical & Ethical Disclaimer

> **IMPORTANT:** This software is developed for research, academic, and clinical triage assistance purposes only. It is **not** an FDA/CE-cleared medical device and must **not** replace the independent diagnostic judgment of a licensed dental clinician or radiologist. Always correlate algorithmic predictions with direct clinical examination and patient history.

---

## 📄 License & Contributing

- **License:** Distributed under the [MIT License](LICENSE).
- **Contributions:** We welcome PRs! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code standards and tests.
