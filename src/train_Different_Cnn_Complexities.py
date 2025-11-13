import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import time
from datetime import datetime
from pathlib import Path

from src.data import make_loaders
from src.transforms import get_transforms
from src.models import ShallowAutoencoder

# ============================================================
# Logging & Directory Setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("cnn_depth_experiments") / RUN_STAMP
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_log = open(RESULTS_DIR / "training_summary.txt", "w")


# ============================================================
# CNN Builder Function — variable depth
# ============================================================
def build_cnn(num_classes, depth, pretrained_path=None):
    """
    Builds a CNN of given depth.
    Each block: Conv2D → ReLU → MaxPool(2).
    Doubles channels every 2 layers.
    """
    layers = []
    in_channels = 3
    out_channels = 32

    for i in range(depth):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        in_channels = out_channels
        if (i + 1) % 2 == 0 and out_channels < 256:
            out_channels *= 2

    encoder = nn.Sequential(*layers)

    # Classifier head
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_channels * (224 // (2 ** depth)) ** 2, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model = nn.Sequential(encoder, classifier)

    # Optionally load pretrained weights
    if pretrained_path:
        try:
            weights = torch.load(pretrained_path, map_location="cpu")
            model[0][0].load_state_dict(weights)  # first conv only
            logger.info(f"Loaded pretrained encoder weights for depth={depth}")
        except Exception as e:
            logger.warning(f"Could not load pretrained encoder: {e}")

    return model


# ============================================================
# Pretrain Shallow Autoencoder (for weight initialization)
# ============================================================
def pretrain_autoencoder(dl_train, device, save_path):
    logger.info("Pretraining shallow autoencoder for encoder weights...")
    model = ShallowAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for x, _ in tqdm(dl_train, desc=f"AE Epoch {epoch+1}", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg = running_loss / len(dl_train.dataset)
        logger.info(f"AE Epoch {epoch+1} | Recon Loss: {avg:.4f}")
        print(f"AE Epoch {epoch+1} | Recon Loss: {avg:.4f}", file=results_log, flush=True)

    torch.save(model.encoder.state_dict(), save_path)
    logger.info(f"✅ Autoencoder pretraining completed → {save_path}")
    print(f"Autoencoder pretraining completed → {save_path}", file=results_log, flush=True)


# ============================================================
# One-Epoch Training Function
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    loop = tqdm(loader, leave=False)
    logger.info(f"Starting epoch {epoch + 1}")

    for x, y in loop:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        acc = correct / total
        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

    avg_loss = running_loss / total
    acc = correct / total
    logger.info(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")
    print(f"Train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}", file=results_log, flush=True)
    return avg_loss, acc


# ============================================================
# Validation Loop
# ============================================================
def evaluate(model, loader, loss_fn, device, epoch):
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
    avg_loss = running_loss / total
    acc = correct / total
    logger.info(f"Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
    print(f"Val Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}", file=results_log, flush=True)
    return avg_loss, acc


# ============================================================
# Main Experiment Routine
# ============================================================
def run_experiment(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transforms = get_transforms(img_size=224)

    # Load data
    dl_train, dl_val = make_loaders(cfg, transforms)

    # Pretrain AE once
    encoder_path = RESULTS_DIR / "pretrained_encoder.pth"
    pretrain_autoencoder(dl_train, device, encoder_path)

    # Train CNNs with increasing depth
    for depth in [3, 5, 7, 9, 11, 13]:
        logger.info(f"\n{'='*60}\n🚀 Training CNN with depth={depth}\n{'='*60}")
        print(f"\nTraining CNN depth={depth}", file=results_log, flush=True)

        model = build_cnn(cfg["num_classes"], depth, pretrained_path=encoder_path).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(cfg["epochs"]):
            tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device, epoch)
            val_loss, val_acc = evaluate(model, dl_val, loss_fn, device, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = RESULTS_DIR / f"best_depth{depth}_epoch{epoch+1}.pth"
                torch.save(model.state_dict(), ckpt)
                logger.info(f"New best model (depth={depth}) saved at epoch {epoch+1} | Val Acc {val_acc:.4f}")

        total_time = time.time() - start_time
        logger.info(f"✅ Depth={depth} completed in {total_time/60:.2f} min | Best Val Acc={best_val_acc:.4f}")
        print(f"Depth={depth} done | Best Val Acc={best_val_acc:.4f}", file=results_log, flush=True)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    cfg = {
        "split_dir": "data/splits_balanced",
        "img_root": "/home/abdulla.alshehhi/Desktop/AI7102-Google_LandMark_Recognition/data/images",
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-6,
        "weight_decay": 1e-4,
        "epochs": 65,
        "num_classes": 100,
    }

    run_experiment(cfg)
    results_log.write("\nAll depth experiments completed successfully!\n")
    results_log.close()
    logger.info("All depth experiments completed successfully!")
