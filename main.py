"""Unified Command-Line Interface for Dental Vision AI."""

import argparse
import sys
from dental_vision.config import CLINICAL_CATEGORIES, DENTAL_CLASSES


def print_info() -> None:
    """Print project overview, clinical categories, and supported pathologies."""
    print("=" * 70)
    print("[*] Dental Vision AI: Panoramic Radiograph Tooth & Pathology Suite")
    print("=" * 70)
    print(f"\nSupported Dental Findings ({len(DENTAL_CLASSES)} Total Classes):")
    for category, classes in CLINICAL_CATEGORIES.items():
        print(f"\n  [+] {category}:")
        for cls_name in classes:
            print(f"     - {cls_name}")
    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dental Vision AI: Panoramic Dental X-ray Tooth & Pathology Detection Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: info
    subparsers.add_parser("info", help="Display clinical taxonomy and model details")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train/fine-tune YOLO model on dental dataset")
    train_parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    train_parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Epoch count")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    train_parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, 0)")
    train_parser.add_argument("--project", type=str, default="runs/train", help="Save directory")
    train_parser.add_argument("--name", type=str, default="dental_run", help="Run name")
    train_parser.add_argument("--fraction", type=float, default=None, help="Subset fraction (e.g. 0.1)")

    # Command: predict
    predict_parser = subparsers.add_parser("predict", help="Run inference on dental X-rays")
    predict_parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to .pt weights (default: weights/best.pt)")
    predict_parser.add_argument("--source", type=str, required=True, help="Path to image or directory")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    predict_parser.add_argument("--output", type=str, default="output/prediction.jpg", help="Output path")
    predict_parser.add_argument("--report", type=str, default=None, help="Save JSON report")

    # Command: export
    export_parser = subparsers.add_parser("export", help="Export model to ONNX/TorchScript/TensorRT")
    export_parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to .pt weights (default: weights/best.pt)")
    export_parser.add_argument("--format", type=str, default="onnx", help="Target format (onnx, torchscript)")
    export_parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "info":
        print_info()

    elif args.command == "train":
        from dental_vision.train import DentalTrainer
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

    elif args.command == "predict":
        from dental_vision.predict import DentalPredictor
        predictor = DentalPredictor(weights_path=args.weights, conf=args.conf)
        predictor.annotate_and_save(args.source, args.output, conf=args.conf)
        report = predictor.generate_report(args.source, conf=args.conf, output_json=args.report)
        print(f"\n✅ Total Findings Detected: {report['total_findings']}")
        for cls_name, count in report["class_summary"].items():
            print(f"  • {cls_name}: {count}")

    elif args.command == "export":
        from dental_vision.model import DentalModel
        model = DentalModel(args.weights)
        model.export(format=args.format, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
