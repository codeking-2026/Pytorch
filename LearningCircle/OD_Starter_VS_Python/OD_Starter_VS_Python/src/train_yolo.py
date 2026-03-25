
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    model.train(data="data.yaml", epochs=20, imgsz=640, batch=16, lr0=0.01, weight_decay=0.0005, device=device)
    model.val()
    model.predict("data/images/val", save=True)
    print("Done. See 'runs' folder.")
