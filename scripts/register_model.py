from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID = "cb7fa9472dfa483fb2807472d5f35fa6"
MODEL_NAME = "Construction-PPE-YOLO11n"

WEIGHTS = Path(
    "runs/lab04_script/yolo11n_mlflow_exp01/weights/best.pt"
).resolve()


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    with mlflow.start_run(run_id=RUN_ID):
        model_info = mlflow.pyfunc.log_model(
            name="yolo_model",
            python_model="scripts/yolo_mlflow_model.py",
            artifacts={"weights": str(WEIGHTS)},
            input_example=pd.DataFrame({
                "image": [str(Path("bus.jpg").resolve())]
            })
        )

    version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        MODEL_NAME,
        "candidate",
        version.version
    )

    print("Model registered successfully")
    print(f"Model name: {MODEL_NAME}")
    print(f"Model version: {version.version}")
    print(f"Model URI: {model_info.model_uri}")
    print("Model alias: candidate")


if __name__ == "__main__":
    main()
