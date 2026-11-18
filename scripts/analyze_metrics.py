
from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser(
        description="Analyze YOLO training metrics"
    )

    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/lab03/metrics"
    )

    return parser.parse_args()


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def find_column(df, required_parts):
    for column in df.columns:
        if all(part.lower() in column.lower()
               for part in required_parts):
            return column

    raise KeyError(f"Column not found: {required_parts}")


def build_metric_summary(df):
    columns = {
        "epoch": "epoch",
        "precision": find_column(df, ["precision"]),
        "recall": find_column(df, ["recall"]),
        "mAP50": find_column(df, ["map50"]),
        "mAP50_95": find_column(df, ["map50-95"])
    }

    if columns["mAP50"] == columns["mAP50_95"]:
        columns["mAP50"] = find_column(df, ["map50("])

    best_row = df.loc[df[columns["mAP50_95"]].idxmax()]
    last_row = df.iloc[-1]

    def extract_metrics(row):
        return {
            key: float(row[column])
            for key, column in columns.items()
            if key != "epoch"
        }

    return {
        "best_epoch_by_map50_95": int(best_row["epoch"]),
        "last_epoch": int(last_row["epoch"]),
        "best_metrics": extract_metrics(best_row),
        "last_epoch_metrics": extract_metrics(last_row),
        "columns": columns
    }


def save_summary(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics_summary.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    rows = [
        {
            "stage": "best_epoch_by_mAP50_95",
            "epoch": summary["best_epoch_by_map50_95"],
            **summary["best_metrics"]
        },
        {
            "stage": "last_epoch",
            "epoch": summary["last_epoch"],
            **summary["last_epoch_metrics"]
        }
    ]

    csv_path = output_dir / "metrics_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved metric summary to: {json_path}")
    print(f"Saved metric table to: {csv_path}")


def plot_main_metrics(df, summary, output_dir):
    columns = summary["columns"]

    plots = [
        (
            ["precision", "recall"],
            "Precision and Recall during training",
            "precision_recall_plot.png"
        ),
        (
            ["mAP50", "mAP50_95"],
            "mAP metrics during training",
            "map_plot.png"
        )
    ]

    for metrics, title, filename in plots:
        plt.figure(figsize=(10, 6))

        for metric in metrics:
            plt.plot(
                df[columns["epoch"]],
                df[columns[metric]],
                marker="o",
                label=metric
            )

        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

    print(f"Saved metric plots to: {output_dir}")


def main():
    args = parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    df = normalize_columns(pd.read_csv(results_path))

    summary = build_metric_summary(df)

    save_summary(summary, output_dir)
    plot_main_metrics(df, summary, output_dir)

    print("\nMetric analysis completed")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
