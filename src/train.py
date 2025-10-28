# src/train.py
import torch
from torch import nn, optim
from tqdm import tqdm
from src.data import make_loaders
from src.transforms import get_transforms
from torchvision import models

# ------------------------------
# 1. Define models
# ------------------------------

class ShallowCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in [64, 128, 256, 512]:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_vit(num_classes):
    vit = models.vit_b_16(weights="IMAGENET1K_V1")
    vit.heads = nn.Linear(vit.heads.head.in_features, num_classes)
    return vit

# ------------------------------
# 2. Training utilities
# ------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total


# ------------------------------
# 3. Main training loop
# ------------------------------

def train_model(model, cfg, dl_train, dl_val, device):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, dl_val, loss_fn, device)
        print(f"Epoch {epoch+1}/{cfg['epochs']}: "
              f"Train Acc {tr_acc:.3f} | Val Acc {val_acc:.3f} | "
              f"Train Loss {tr_loss:.3f} | Val Loss {val_loss:.3f}")

# ------------------------------
# 4. Run experiments
# ------------------------------

if __name__ == "__main__":
    cfg = {
        "split_dir": "data/splits_80_10_10/merged_full",
        "img_root": "/Users/abdullaalbudoor/Desktop/raw/train_images_shards",
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 15,
        "num_classes": 561,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = get_transforms(img_size=128)
    dl_train, dl_val = make_loaders(cfg, transforms)

    print("Training Shallow CNN")
    model = ShallowCNN(num_classes=cfg["num_classes"]).to(device)
    train_model(model, cfg, dl_train, dl_val, device)

    print("\nTraining Deep CNN")
    model = DeepCNN(num_classes=cfg["num_classes"]).to(device)
    train_model(model, cfg, dl_train, dl_val, device)

    print("\nTraining Vision Transformer (ViT)")
    model = get_vit(num_classes=cfg["num_classes"]).to(device)
    train_model(model, cfg, dl_train, dl_val, device)
