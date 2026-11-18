import mlflow
import pandas as pd

from mlflow.pyfunc import PythonModel
from ultralytics import YOLO


class YOLOModel(PythonModel):

    def load_context(self, context):
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        outputs = []

        for image in model_input["image"]:
            result = self.model.predict(
                source=image,
                imgsz=416,
                conf=0.25,
                device="cpu",
                verbose=False
            )[0]

            outputs.append(result.to_json())

        return pd.DataFrame({"detections": outputs})


mlflow.models.set_model(YOLOModel())
