
from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(
        description="YOLO object detection inference"
    )

    parser.add_argument(
        "--model",
        default="runs/lab03_script/yolo11n_construction_ppe_small/weights/best.pt"
    )

    parser.add_argument(
        "--source",
        default="datasets/construction-ppe-small/images/test"
    )

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="artifacts/lab03")
    parser.add_argument("--name", default="predictions")

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model).resolve()
    source_path = Path(args.source).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True
    )

    total_objects = sum(len(result.boxes) for result in results)

    print("\nPrediction completed successfully!")
    print("Processed images:", len(results))
    print("Detected objects:", total_objects)
    print("Results directory:", results[0].save_dir)


if __name__ == "__main__":
    main()
