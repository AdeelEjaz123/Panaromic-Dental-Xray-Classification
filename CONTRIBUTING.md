# Contributing to Dental Vision AI

Thank you for your interest in contributing to **Dental Vision AI**! We welcome contributions from researchers, clinicians, deep learning engineers, and open-source enthusiasts.

---

## Code of Conduct
Please be respectful, collaborative, and inclusive. We aim to foster an open, welcoming environment.

---

## How Can You Contribute?
1. **Reporting Bugs**: Open an issue detailing your OS, Python version, PyTorch/CUDA version, and stack trace.
2. **Feature Requests**: Suggest architectural enhancements (e.g., YOLOv9/v10/v11 integrations, segmentation heads, DICOM support).
3. **Model Weights & Datasets**: Benchmark new dental datasets (e.g., Tufts Dental, UFBA_UESC_DENTAL, panoramic pathology benchmarks).
4. **Code Improvements**: Fix lint errors, optimize data augmentation, improve visualization utilities.

---

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/dental-xray-classification.git
   cd dental-xray-classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies in editable mode:
   ```bash
   pip install -e .
   pip install pytest flake8 black
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

---

## Pull Request Guidelines
- Follow PEP 8 style conventions.
- Format with `black` and check with `flake8`.
- Ensure all unit tests pass before submitting your PR.
- Add descriptive commit messages adhering to Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`).
