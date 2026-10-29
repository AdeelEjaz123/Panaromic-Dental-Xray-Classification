"""Inference Engine & Clinical Triage Report Generator for Dental Vision."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO

from dental_vision.config import CLINICAL_CATEGORIES, DEFAULT_CONFIG, get_class_color


class DentalPredictor:
    """Predictor engine for panoramic dental radiograph inference and pathology detection."""

    def __init__(
        self,
        weights_path: str = "weights/best.pt",
        conf: float = DEFAULT_CONFIG["conf"],
        iou: float = DEFAULT_CONFIG["iou"],
        device: Optional[Union[str, int]] = None,
    ):
        """Initialize the predictor.

        Args:
            weights_path: Path to trained YOLO weights file (.pt).
            conf: Confidence threshold (default: 0.25).
            iou: NMS IoU threshold (default: 0.45).
            device: Computation device.
        """
        resolved_path = weights_path
        if not os.path.exists(resolved_path):
            if os.path.exists("weights/best.pt"):
                resolved_path = "weights/best.pt"
            elif os.path.exists("run_1-2/weights/best.pt"):
                resolved_path = "run_1-2/weights/best.pt"

        self.weights_path = resolved_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(self.weights_path)

    def _map_category(self, class_name: str) -> str:
        """Map class name to higher-level clinical triage category."""
        for category, classes in CLINICAL_CATEGORIES.items():
            if class_name.lower() in [c.lower() for c in classes]:
                return category
        return "General Anatomy"

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run model inference on single image or array.

        Args:
            source: File path or numpy image array (BGR/RGB).
            conf: Override default confidence threshold.
            iou: Override default IoU threshold.

        Returns:
            List of detected findings containing class, confidence, bbox, category.
        """
        threshold = conf if conf is not None else self.conf
        iou_thresh = iou if iou is not None else self.iou

        results = self.model.predict(
            source=source,
            conf=threshold,
            iou=iou_thresh,
            device=self.device,
            verbose=False,
        )

        findings = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"Class_{cls_id}")
                confidence = float(box.conf[0].item())
                xyxy = [round(coord, 2) for coord in box.xyxy[0].tolist()]

                findings.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(confidence, 4),
                        "box_xyxy": xyxy,
                        "clinical_category": self._map_category(cls_name),
                    }
                )

        return findings

    def generate_report(
        self,
        source_path: Union[str, Path],
        conf: Optional[float] = None,
        output_json: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate structured clinical triage JSON report for an X-ray.

        Args:
            source_path: Path to dental X-ray image.
            conf: Confidence threshold.
            output_json: Path to save report JSON.

        Returns:
            Dictionary containing metadata, summary counts, and detailed findings.
        """
        path_str = str(Path(source_path).resolve())
        findings = self.predict(path_str, conf=conf)

        # Compute summary metrics by class and clinical category
        class_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        for item in findings:
            c_name = item["class_name"]
            cat = item["clinical_category"]
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1

        report = {
            "source_image": Path(source_path).name,
            "total_findings": len(findings),
            "confidence_threshold": conf or self.conf,
            "category_summary": category_counts,
            "class_summary": class_counts,
            "findings": findings,
        }

        if output_json:
            out_p = Path(output_json).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Report]  Clinical report saved to: {out_p}")

        return report

    def annotate_and_save(
        self,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        line_thickness: int = 2,
    ) -> str:
        """Annotate X-ray with colored bounding boxes & labels, and save to disk."""
        findings = self.predict(source_path, conf=conf, iou=iou)
        img = cv2.imread(str(source_path))
        if img is None:
            raise ValueError(f"Could not read image from {source_path}")

        for item in findings:
            x1, y1, x2, y2 = map(int, item["box_xyxy"])
            label = f"{item['class_name']} {item['confidence']:.2f}"
            color = get_class_color(item["class_name"])

            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            # Label background badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), img)
        print(f"[Saved]  Annotated image saved to: {out_file}")
        return str(out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dental pathology inference on panoramic X-rays."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights (.pt file, defaults to weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to X-ray image or folder of images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/prediction.jpg",
        help="Path to save annotated output image",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save clinical JSON report (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DentalPredictor(weights_path=args.weights, conf=args.conf)

    source_path = Path(args.source)
    if source_path.is_file():
        predictor.annotate_and_save(args.source, args.output, conf=args.conf)
        report = predictor.generate_report(args.source, conf=args.conf, output_json=args.report)
        print("\n=== Findings Summary ===")
        print(f"Total Conditions Detected: {report['total_findings']}")
        for cls, count in report["class_summary"].items():
            print(f"  - {cls}: {count}")
    elif source_path.is_dir():
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in source_path.iterdir() if f.suffix.lower() in exts]
        print(f"[Batch]  Processing batch of {len(images)} images in {source_path}...")
        for img_path in images:
            save_name = out_dir / f"pred_{img_path.name}"
            predictor.annotate_and_save(img_path, save_name, conf=args.conf)


if __name__ == "__main__":
    main()
