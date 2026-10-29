"""Test dataset utilities."""

import tempfile
from pathlib import Path
from dental_vision.dataset import DentalDataset


def test_create_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dummy_dataset"
        dataset_dir.mkdir()
        yaml_out = Path(tmpdir) / "output.yaml"

        generated_path = DentalDataset.create_yaml(
            dataset_dir=dataset_dir,
            output_yaml_path=yaml_out,
            class_names=["Tooth", "Caries", "Filling"],
        )

        assert Path(generated_path).exists()
        with open(generated_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Tooth" in content
            assert "Caries" in content
            assert "Filling" in content
            assert "nc: 3" in content
