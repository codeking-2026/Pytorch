
import os, glob
from PIL import Image, ImageDraw
import torch, torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

CLASSES  = ["box", "ball"]
NUM_CLASSES = len(CLASSES) + 1

ROOT = "./data"
TRAIN_IM = os.path.join(ROOT, "images/train")
TRAIN_LB = os.path.join(ROOT, "labels/train")
VAL_IM   = os.path.join(ROOT, "images/val")
VAL_LB   = os.path.join(ROOT, "labels/val")

def read_yolo_txt(txt_path):
    if not os.path.exists(txt_path): return []
    lines = open(txt_path, "r", encoding="utf-8").read().strip().splitlines()
    items = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) != 5: 
            continue
        c, xc, yc, w, h = p
        items.append((int(c), float(xc), float(yc), float(w), float(h)))
    return items

def yolo_to_xyxy(xc, yc, w, h, W, H):
    bw, bh = w * W, h * H
    cx, cy = xc * W, yc * H
    x1, y1 = cx - bw/2, cy - bh/2
    x2, y2 = cx + bw/2, cy + bh/2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    return [x1, y1, x2, y2]

class YoloDetectionDataset(Dataset):
    def __init__(self, im_dir, lb_dir):
        self.im_paths = sorted([p for p in glob.glob(os.path.join(im_dir, "*.*")) 
                                if p.lower().endswith((".jpg",".jpeg",".png"))])
        self.lb_dir = lb_dir
        self.tf = T.ToTensor()

    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        img = Image.open(im_path).convert("RGB")
        W, H = img.size
        base = os.path.splitext(os.path.basename(im_path))[0]
        lb_path = os.path.join(self.lb_dir, base + ".txt")
        anns = read_yolo_txt(lb_path)
        boxes, labels = [], []
        for c, xc, yc, w, h in anns:
            x1,y1,x2,y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            if x2-x1>1 and y2-y1>1:
                boxes.append([x1,y1,x2,y2])
                labels.append(c+1)   # shift by +1 (0 = background)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return self.tf(img), target

def collate_fn(batch):
    imgs, tgts = list(zip(*batch))
    return list(imgs), list(tgts)

def main():
    train_ds = YoloDetectionDataset(TRAIN_IM, TRAIN_LB)
    val_ds   = YoloDetectionDataset(VAL_IM, VAL_LB)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feat, NUM_CLASSES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    def run_epoch(loader, train=True):
        if train: model.train()
        else:     model.eval()
        total = 0.0
        for imgs, tgts in loader:
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            if train:
                losses = model(imgs, tgts)
                loss = sum(losses.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss)
            else:
                with torch.no_grad():
                    losses = model(imgs, tgts)
                    loss = sum(losses.values())
                    total += float(loss)
        return total / max(1, len(loader))

    EPOCHS = 10
    for ep in range(1, EPOCHS+1):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl,   train=False)
        lr_sched.step()
        print(f"Epoch {ep:02d} | train loss {tr:.4f} | val loss {va:.4f}")

    # Save a few predictions
    os.makedirs("preds", exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(val_dl):
            imgs = [img.to(device) for img in imgs]
            outs = model(imgs)
            for j, (img, out) in enumerate(zip(imgs, outs)):
                pil = T.ToPILImage()(img.cpu())
                draw = ImageDraw.Draw(pil)
                for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
                    if s < 0.5: continue
                    x1,y1,x2,y2 = b.tolist()
                    draw.rectangle([x1,y1,x2,y2], width=3, outline="red")
                    cls_name = CLASSES[l.item()-1] if 1 <= l.item() <= len(CLASSES) else str(l.item())
                    draw.text((x1, y1), f"{cls_name} {s:.2f}", fill="red")
                pil.save(f"preds/ep{EPOCHS}_sample{i}_{j}.jpg")
            if i >= 2:
                break
    print("Saved predictions to preds/*.jpg")

if __name__ == "__main__":
    main()
