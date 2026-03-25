
from ultralytics import YOLO
import torch

# Train YOLOv8 on the included synthetic dataset
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # tiny and fast; switch to yolov8s.pt for better accuracy
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.01,
        weight_decay=0.0005,
        device=device
    )
    # Validate and run a quick prediction on val images
    model.val()
    model.predict("data/images/val", save=True)
    print("Training complete. Check the 'runs' directory for logs and predictions.")
