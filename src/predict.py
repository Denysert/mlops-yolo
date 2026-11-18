from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="bus.jpg",
    conf=0.25,
    device="cpu",
    save=True
)

print("Prediction completed successfully.")
