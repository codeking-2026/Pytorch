
# Object Detection Starter Kit (PyTorch + YOLO)

This package contains a tiny **synthetic** object-detection dataset (two classes: ['box', 'ball'])
and ready-to-run training scripts for **Ultralytics YOLOv8** and **torchvision Faster R-CNN**.

## Folder Layout
```
od_starter_kit/
├─ data/
│  ├─ images/
│  │  ├─ train/   # training images
│  │  └─ val/     # validation images
│  ├─ labels/
│  │  ├─ train/   # YOLO TXT labels
│  │  └─ val/
│  └─ data.yaml   # dataset config for YOLO
├─ train_yolo.py
├─ train_torchvision.py
└─ README.md
```

## Quick Start — YOLOv8
```bash
pip install ultralytics
python train_yolo.py
```

- Results (curves, best.pt) will be saved under `runs/detect/train*`.
- To predict and save annotated images after training, see the bottom of `train_yolo.py`.

## Quick Start — torchvision Faster R-CNN
```bash
pip install torch torchvision pillow
python train_torchvision.py
```
- Predictions will be saved under `preds/` at the end of the script.

---

## Classes
- 0: box (rectangle)
- 1: ball (circle)

This synthetic dataset is for learning purposes — replace images & labels with your own data following the same structure.
