import torch
from torch import nn, optim
from tqdm import tqdm
import logging
import time

from src.data import make_loaders
from src.transforms import get_transforms
from src.models import (
    ShallowCNN,
    DeepCNNPretrained,
    ViTModel,
    ShallowAutoencoder
)

results_log = open("training_summary.txt", "w")

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Utility: One epoch training
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, log_interval=50):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    loop = tqdm(loader, leave=False)
    logger.info(f"Starting epoch {epoch + 1}")

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # Optional live accuracy in tqdm bar
        acc = correct / total if total > 0 else 0
        loop.set_postfix(loss=loss.item(), acc=f"{acc:.3f}")

        # Log periodically
        if (batch_idx + 1) % log_interval == 0:
            logger.info(f"Batch {batch_idx+1:04d}/{len(loader)} | "
                        f"Train Loss: {running_loss/total:.4f} | Acc: {acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    logger.info(f"✅ Epoch {epoch+1} Summary | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
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
    accuracy = correct / total
    logger.info(f"🧪 Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# ---------------------------------------------------------
# Pretraining for Shallow Autoencoder
# ---------------------------------------------------------
def pretrain_autoencoder(cfg, dl_train, device):
    logger.info("🔧 Pretraining Shallow Autoencoder...")

    print("\n==============================", file=results_log)
    print("🔧 Pretraining Shallow Autoencoder", file=results_log)
    print("==============================", file=results_log)

    model = ShallowAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        model.train()
        running_loss = 0
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

    torch.save(model.encoder.state_dict(), "shallow_encoder_pretrained.pth")
    logger.info("✅ Autoencoder pretraining completed and weights saved.")
    print("✅ Autoencoder pretraining completed and weights saved.\n", file=results_log, flush=True)


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def train_model(model, cfg, dl_train, dl_val, device):
    logger.info("🚀 Starting fine-tuning loop...")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device, epoch)
        val_loss, val_acc = evaluate(model, dl_val, loss_fn, device, epoch)

        logger.info(
            f"📊 Epoch {epoch+1}/{cfg['epochs']} | "
            f"Train Acc {tr_acc:.3f} | Val Acc {val_acc:.3f} | "
            f"Train Loss {tr_loss:.3f} | Val Loss {val_loss:.3f}"
        )

        print(f"Epoch {epoch+1} Summary | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", file=results_log, flush=True)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | Train Acc {tr_acc:.3f} | Val Acc {val_acc:.3f} | Train Loss {tr_loss:.3f} | Val Loss {val_loss:.3f}", file=results_log, flush=True)

        # Track best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pth")
            logger.info(f"🌟 New best model saved at epoch {epoch+1} with Val Acc {val_acc:.4f}")

    total_time = time.time() - start_time
    logger.info(f"✅ Fine-tuning completed in {total_time/60:.2f} min | Best Val Acc: {best_val_acc:.4f}")

    print(f"\n✅ Fine-tuning completed in {total_time/60:.2f} min | Best Val Acc: {best_val_acc:.4f}", file=results_log, flush=True)
    print("="*80, file=results_log)


# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
if __name__ == "__main__":
    cfg = {
        "split_dir": "data/splits_100",
        "img_root": "/home/abdulla.alshehhi/Desktop/AI7102-Google_LandMark_Recognition/data/images",
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-6,
        "weight_decay": 1e-4,
        "epochs": 65,
        "num_classes": 100,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transforms = get_transforms(img_size=224)
    logger.info("Loading DataLoaders...")
    dl_train, dl_val = make_loaders(cfg, transforms)
    logger.info("✅ DataLoaders ready")

    # -----------------------------------------------------
    # 1️⃣ Pretrain Shallow Autoencoder (once)
    # -----------------------------------------------------
    pretrain_autoencoder(cfg, dl_train, device)

    # -----------------------------------------------------
    # 2️⃣ Train Shallow CNN (with pretrained encoder)
    # -----------------------------------------------------
    print("\n==============================", file=results_log)
    print("🚀 Training Shallow CNN (Pretrained)", file=results_log)
    print("==============================", file=results_log)


    logger.info("\nTraining Shallow CNN (Pretrained)")
    model = ShallowCNN(cfg["num_classes"], pretrained_path="shallow_encoder_pretrained.pth").to(device)
    train_model(model, cfg, dl_train, dl_val, device)

    # -----------------------------------------------------
    # 3️⃣ Train Deep CNN (ResNet backbone)
    # -----------------------------------------------------

    print("\n==============================", file=results_log)
    print("🚀 Training Deep CNN (ResNet Pretrained)", file=results_log)
    print("==============================", file=results_log)

    logger.info("\nTraining Deep CNN (ResNet Pretrained)")
    model = DeepCNNPretrained(cfg["num_classes"], pretrained=True).to(device)
    train_model(model, cfg, dl_train, dl_val, device)

    # -----------------------------------------------------
    # 4️⃣ Train Vision Transformer (ViT Pretrained)
    # -----------------------------------------------------

    print("\n==============================", file=results_log)
    print("🚀 Training Vision Transformer (ViT-B16 Pretrained)", file=results_log)
    print("==============================", file=results_log)
    logger.info("\nTraining Vision Transformer (ViT-B16 Pretrained)")
    model = ViTModel(cfg["num_classes"], pretrained=True).to(device)
    train_model(model, cfg, dl_train, dl_val, device)

    results_log.write("🎯 All experiments completed successfully!\n")
    results_log.close()
    
    logger.info("🎯 All experiments completed successfully!")
