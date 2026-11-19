
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Batch Inference"
    )

    parser.add_argument(
        "--model", default="/app/models/best.pt"
    )
    parser.add_argument(
        "--source", default="/app/input"
    )
    parser.add_argument(
        "--output", default="/app/output"
    )
    parser.add_argument(
        "--imgsz", type=int, default=416
    )
    parser.add_argument(
        "--device", default="cpu"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25
    )

    return parser.parse_args()


def validate_inputs(model_path, source_path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source not found: {source_path}"
        )

    if source_path.is_file():
        images = [source_path] if (
            source_path.suffix.lower() in IMAGE_EXTENSIONS
        ) else []
    else:
        images = sorted(
            path for path in source_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not images:
        raise ValueError("No supported images found")

    return images


def main():
    args = parse_args()

    model_path = Path(args.model).resolve()
    source_path = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    images = validate_inputs(model_path, source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("YOLO Batch Inference")
    print(f"Model: {model_path}")
    print(f"Source: {source_path}")
    print(f"Output: {output_dir}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")

    start = time.perf_counter()
    model = YOLO(str(model_path))
    model_loading_seconds = time.perf_counter() - start

    start = time.perf_counter()

    results = model.predict(
        source=[str(image) for image in images],
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True
    )

    inference_seconds = time.perf_counter() - start

    images_processed = len(results)

    detections = sum(
        len(result.boxes)
        for result in results
        if result.boxes is not None
    )

    average_model_inference_ms = (
        sum(result.speed["inference"] for result in results)
        / images_processed
    ) if images_processed else 0.0

    throughput = (
        images_processed / inference_seconds
        if inference_seconds > 0 else 0.0
    )

    summary = {
        "model": str(model_path),
        "source": str(source_path),
        "output": str(output_dir),
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "images_processed": images_processed,
        "detections": detections,
        "model_loading_seconds": round(
            model_loading_seconds, 4
        ),
        "inference_seconds": round(
            inference_seconds, 4
        ),
        "average_model_inference_ms": round(
            average_model_inference_ms, 4
        ),
        "throughput_images_per_second": round(
            throughput, 4
        )
    }

    summary_path = output_dir / "inference_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\nInference completed")
    print(f"Images processed: {images_processed}")
    print(f"Detections: {detections}")
    print(f"Model loading: {model_loading_seconds:.4f} s")
    print(f"Inference job: {inference_seconds:.4f} s")
    print(f"Throughput: {throughput:.4f} images/s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
