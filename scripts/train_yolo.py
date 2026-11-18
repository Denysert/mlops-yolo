from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import platform
import sys
import os
import math
import subprocess

import pandas as pd
import mlflow
import yaml

from ultralytics import YOLO, settings


def parse_args():
    parser = ArgumentParser(
        description="Automated YOLO training with MLflow"
    )

    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="datasets/construction-ppe-small/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/lab04_script")
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=10)

    return parser.parse_args()


def create_experiment_name(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = Path(model_name).stem
    return f"{clean_model_name}_construction_ppe_{timestamp}"


def git_command(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


def get_dataset_revision():
    dvc_path = Path("datasets/construction-ppe-small.dvc")

    if not dvc_path.is_file():
        return "unknown"

    data = yaml.safe_load(dvc_path.read_text())
    return data["outs"][0]["md5"]


def save_run_metadata(run_dir, args, run_id):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        **vars(args),
        "mlflow_run_id": run_id,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "git_commit": git_command("rev-parse", "HEAD"),
        "dataset_revision": get_dataset_revision(),
        "run_dir": str(run_dir),
        "best_model_path": str(run_dir / "weights" / "best.pt"),
        "last_model_path": str(run_dir / "weights" / "last.pt")
    }

    output_path = run_dir / "run_metadata.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def log_yolo_metrics(run_dir):
    results_path = run_dir / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    metric_columns = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50_95": "metrics/mAP50-95(B)"
    }

    count = 0

    for _, row in df.iterrows():
        epoch = int(row["epoch"])

        for metric_name, column in metric_columns.items():
            value = float(row[column])

            if math.isfinite(value):
                mlflow.log_metric(
                    metric_name,
                    value,
                    step=epoch
                )
                count += 1

    print(f"Logged metric values: {count}")


def log_yolo_artifacts(run_dir):
    files = {
        "results.csv": "metrics",
        "results.png": "plots",
        "confusion_matrix.png": "plots",
        "confusion_matrix_normalized.png": "plots",
        "BoxF1_curve.png": "plots",
        "BoxPR_curve.png": "plots",
        "BoxP_curve.png": "plots",
        "BoxR_curve.png": "plots",
        "args.yaml": "metadata",
        "run_metadata.json": "metadata",
        "weights/best.pt": "model",
        "weights/last.pt": "model"
    }

    count = 0

    for filename, destination in files.items():
        path = run_dir / filename

        if path.is_file():
            mlflow.log_artifact(
                str(path),
                artifact_path=destination
            )

            print(f"Logged artifact: {filename}")
            count += 1

    print(f"Total artifacts: {count}")


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if args.name is None:
        args.name = create_experiment_name(args.model)

    settings.update({"mlflow": False})

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Construction-PPE-YOLO-Automated")

    print("Starting YOLO training with MLflow")
    print(f"Tracking URI: {tracking_uri}")

    with mlflow.start_run(run_name=args.name) as run:

        run_id = run.info.run_id

        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "seed": 0
        })

        git_status = git_command("status", "--porcelain")

        mlflow.set_tags({
            "dataset.name": "Construction-PPE-small",
            "dataset.revision": get_dataset_revision(),
            "git.commit": git_command("rev-parse", "HEAD"),
            "git.branch": git_command("rev-parse", "--abbrev-ref", "HEAD"),
            "git.dirty": (
                "unknown" if git_status == "unknown"
                else str(bool(git_status)).lower()
            ),
            "model.family": "YOLO11",
            "task": "object_detection",
            "environment": "python-script",
            "source": "lab04-script"
        })

        print(f"MLflow Run ID: {run_id}")

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

        save_run_metadata(run_dir, args, run_id)
        log_yolo_metrics(run_dir)
        log_yolo_artifacts(run_dir)

        print("\nTraining completed")
        print(f"Results saved to: {run_dir}")
        print(f"Best model: {run_dir / 'weights' / 'best.pt'}")
        print(f"Last model: {run_dir / 'weights' / 'last.pt'}")
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()
