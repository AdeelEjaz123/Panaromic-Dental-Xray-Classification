"""Visualization utilities: Tooth isolation panels, side-by-side triage, and report plotting."""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

from dental_vision.config import get_class_color


class DentalVisualizer:
    """Visualization suite for dental radiograph findings."""

    @staticmethod
    def create_single_tooth_isolation_grid(
        image_path: Union[str, Path],
        findings: List[Dict[str, Any]],
        output_image_path: Optional[Union[str, Path]] = None,
        cols: int = 5,
        figsize_per_row: int = 4,
    ) -> Optional[plt.Figure]:
        """Generate a clinical grid highlighting each individual detected tooth/finding on full X-ray.

        Args:
            image_path: Path to original X-ray radiograph.
            findings: List of detection dictionaries containing 'box_xyxy', 'class_name', 'confidence'.
            output_image_path: Destination path to save the generated grid image.
            cols: Number of columns in subplot grid.
            figsize_per_row: Height multiplier per row.

        Returns:
            Matplotlib Figure object if generated.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        total_findings = len(findings)
        if total_findings == 0:
            print("[Warning]  No findings detected to plot in isolation grid.")
            return None

        rows = math.ceil(total_findings / cols)
        fig = plt.figure(figsize=(20, figsize_per_row * rows))

        for idx, item in enumerate(findings):
            img_copy = img_rgb.copy()
            x1, y1, x2, y2 = map(int, item["box_xyxy"])
            label = f"{item['class_name']} #{idx+1} ({item['confidence']:.2f})"

            # Highlight bounding box in vibrant red/class color
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(
                img_copy,
                label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img_copy)
            ax.set_title(label, fontsize=10)
            ax.axis("off")

        plt.tight_layout()

        if output_image_path:
            out_p = Path(output_image_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), bbox_inches="tight", dpi=150)
            print(f"[Tooth]  Tooth isolation grid saved to: {out_p}")

        return fig

    @staticmethod
    def side_by_side_comparison(
        raw_image_path: Union[str, Path],
        annotated_image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Combine raw and annotated radiograph into a side-by-side diagnostic panel."""
        raw = cv2.imread(str(raw_image_path))
        annotated = cv2.imread(str(annotated_image_path))

        if raw is None or annotated is None:
            raise ValueError("Failed to load one or both comparison images.")

        if raw.shape != annotated.shape:
            annotated = cv2.resize(annotated, (raw.shape[1], raw.shape[0]))

        combined = np.hstack([raw, annotated])

        # Add headers
        header = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
        w_half = combined.shape[1] // 2
        cv2.putText(
            header,
            "RAW PANORAMIC RADIOGRAPH",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            "AI PATHOLOGY & TOOTH DETECTION",
            (w_half + 20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        final_panel = np.vstack([header, combined])

        if output_path:
            out_p = Path(output_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), final_panel)
            print(f"[Saved]  Side-by-side panel saved to: {out_p}")

        return final_panel
