
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import platform
import sys

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(
        description="Automated YOLO training"
    )

    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="datasets/construction-ppe-small/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/lab03_script")
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=10)

    return parser.parse_args()


def create_experiment_name(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = Path(model_name).stem
    return f"{clean_model_name}_construction_ppe_{timestamp}"


def save_run_metadata(run_dir, args):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        **vars(args),
        "run_dir": str(run_dir),
        "best_model_path": str(run_dir / "weights" / "best.pt"),
        "last_model_path": str(run_dir / "weights" / "last.pt")
    }

    output_path = run_dir / "run_metadata.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if args.name is None:
        args.name = create_experiment_name(args.model)

    print("Starting YOLO training")

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        patience=args.patience,
        plots=True,
        seed=0,
        deterministic=True
    )

    run_dir = Path(model.trainer.save_dir)
    save_run_metadata(run_dir, args)

    print("\nTraining completed")
    print(f"Results saved to: {run_dir}")
    print(f"Best model: {run_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {run_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
