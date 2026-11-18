
from argparse import ArgumentParser
from pathlib import Path
import shutil


def parse_args():
    parser = ArgumentParser(
        description="Prepare YOLO artifacts for DVC"
    )

    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/lab03"
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=5
    )

    return parser.parse_args()


def copy_file(src, dst):
    if not src.is_file():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Already prepared: {dst}")


def copy_prediction_examples(
    prediction_dir, output_dir, max_predictions
):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(prediction_dir.glob("*.jpg")) +
        list(prediction_dir.glob("*.jpeg")) +
        list(prediction_dir.glob("*.png"))
    )

    selected_images = image_paths[:max_predictions]

    for image_path in selected_images:
        copy_file(image_path, output_dir / image_path.name)

    print(f"Prepared {len(selected_images)} prediction examples")


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)

    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    if not prediction_dir.exists():
        raise FileNotFoundError(prediction_dir)

    model_dir = output_dir / "model"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metadata_dir = output_dir / "metadata"
    prediction_output_dir = output_dir / "predictions"

    copy_file(
        run_dir / "weights/best.pt",
        model_dir / "best.pt"
    )

    copy_file(
        run_dir / "weights/last.pt",
        model_dir / "last.pt"
    )

    copy_file(
        run_dir / "results.csv",
        metrics_dir / "results.csv"
    )

    plot_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png"
    ]

    for filename in plot_files:
        copy_file(
            run_dir / filename,
            plots_dir / filename
        )

    copy_file(
        run_dir / "args.yaml",
        metadata_dir / "args.yaml"
    )

    copy_file(
        run_dir / "run_metadata.json",
        metadata_dir / "run_metadata.json"
    )

    copy_prediction_examples(
        prediction_dir,
        prediction_output_dir,
        args.max_predictions
    )

    print("\nArtifact preparation completed")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
