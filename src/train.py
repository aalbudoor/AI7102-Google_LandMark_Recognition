import torch
from torch import nn, optim
from tqdm import tqdm
import logging
import time
from datetime import datetime
from pathlib import Path

from src.data import make_loaders
from src.transforms import get_transforms
from src.models import (ShallowCNN, DeepCNNPretrained, ViTModel, ShallowAutoencoder)


# Basic logging + results folder setup
# Every run gets its own folder under /runs/YYYYMMDD_HHMMSS/

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("runs") / RUN_STAMP
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_log = open(RESULTS_DIR / "training_summary.txt", "w")


# ---------------------------------------------------------
# TRAIN ONE EPOCH (Mini-batch training loop)
# What happens:
#   - model forward pass
#   - calculate loss
#   - backprop
#   - update weights
# Batch accuracy + loss are tracked live.
# ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, log_interval=50):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    loop = tqdm(loader, leave=False)
    logger.info(f"Starting epoch {epoch + 1}")



    for batch_idx, (x, y) in enumerate(loop):
        # Mini-batch from dataloader
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x) # Forward pass
        loss = loss_fn(logits, y) # Cross-Entropy
        loss.backward() # Backprop
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # live metrics on the progress bar
        acc = correct / total if total > 0 else 0.0
        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")


        # Log every few batches
        if (batch_idx + 1) % log_interval == 0:
            logger.info(f"Batch {batch_idx+1:04d}/{len(loader)} | "
                        f"Train Loss: {running_loss/total:.4f} | Acc: {acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    logger.info(f"Epoch {epoch+1} Summary | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")
    return avg_loss, accuracy





# ---------------------------------------------------------
# VALIDATION LOOP (no backprop)
# Just evaluation: forward → loss → accuracy.
# ---------------------------------------------------------
def evaluate(model, loader, loss_fn, device, epoch):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    logger.info(f"Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# ---------------------------------------------------------
# AUTOENCODER PRETRAINING (only for ShallowCNN)
# Idea:
#   - Train autoencoder to reconstruct images
#   - Use encoder weights as initialization for ShallowCNN
# Very cheap “unsupervised warmup.”
# ---------------------------------------------------------
def pretrain_autoencoder(dl_train, device, epochs=10, lr=1e-3, save_path="shallow_encoder_pretrained.pth"):
    logger.info("Pretraining Shallow Autoencoder...")

    print("\n==============================", file=results_log)
    print("Pretraining Shallow Autoencoder", file=results_log)
    print("==============================", file=results_log)

    model = ShallowAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # reconstructing = MSE


    # A for loop to train AE on unlabeled images
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, _ in tqdm(dl_train, desc=f"AE Epoch {epoch+1}", leave=False):
            x = x.to(device, non_blocking=True)
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
    logger.info(f"Autoencoder pretraining completed and weights saved → {save_path}")
    print(f"Autoencoder pretraining completed and weights saved → {save_path}\n", file=results_log, flush=True)





# ---------------------------------------------------------
# MAIN TRAINING LOOP FOR ANY MODEL (CNN, ResNet, ViT)
# What it does:
#   - train + validate for several epochs
#   - save best checkpoint
# ---------------------------------------------------------
def train_model(model, cfg, dl_train, dl_val, device, model_tag="model", classes_tag="unknown"):
    logger.info(f"🚀 Starting fine-tuning loop... [{model_tag} | {classes_tag} classes]")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW =  optimizer for transformers + modern CNNs
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val_acc = 0.0
    start_time = time.time()



    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device, epoch)
        val_loss, val_acc = evaluate(model, dl_val, loss_fn, device, epoch)

        logger.info(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"Train Acc {tr_acc:.3f} | Val Acc {val_acc:.3f} | "
            f"Train Loss {tr_loss:.3f} | Val Loss {val_loss:.3f}"
        )
        print(
            f"[{model_tag} | {classes_tag}] Epoch {epoch+1}/{cfg['epochs']} | "
            f"Train Acc {tr_acc:.3f} | Val Acc {val_acc:.3f} | "
            f"Train Loss {tr_loss:.3f} | Val Loss {val_loss:.3f}",
            file=results_log, flush=True
        )





        # Save best model based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = RESULTS_DIR / f"best_{model_tag}_{classes_tag}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"New best model saved at epoch {epoch+1} with Val Acc {val_acc:.4f} → {ckpt}")

    total_time = time.time() - start_time
    logger.info(f"Fine-tuning completed in {total_time/60:.2f} min | Best Val Acc: {best_val_acc:.4f}")
    print(
        f"\n[{model_tag} | {classes_tag}] Fine-tuning completed in {total_time/60:.2f} min | "
        f"Best Val Acc: {best_val_acc:.4f}\n" + "="*80,
        file=results_log,
        flush=True
    )






# ---------------------------------------------------------
# FULL EXPERIMENT PIPELINE
# Step 1 → Load 50-class dataset
# Step 2 → Load 75-class dataset
# Step 3 → Load 100-class dataset
# Step 4 → Pretrain AE using the first DataLoader
# Step 5 → Train ShallowCNN, DeepCNN, ViT on all three datasets
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Base cfg; we’ll vary num_classes and CSVs per loop
    cfg = {
        "split_dir": "data/splits_balanced",  # << new balanced CSVs
        "img_root": "/home/abdulla.alshehhi/Desktop/AI7102-Google_LandMark_Recognition/data/images",
        "batch_size": 32,       # mini-batch size
        "num_workers": 4,
        "lr": 1e-6,
        "weight_decay": 1e-4,
        "epochs": 65,
        "num_classes": 100,     # will be overwritten per round
    }

    transforms = get_transforms(img_size=224)

    CLASS_SIZES = [50, 75, 100]
    first_dl_train_for_ae = None



    # A for loop to build all dataloaders once
    for n_cls in CLASS_SIZES:
        logger.info("\n" + "="*80)
        logger.info(f"🎯 Starting training for dataset with {n_cls} classes")
        logger.info("="*80)

        cfg["num_classes"] = n_cls

        # Build loaders for this class count (mini-batch handled by DataLoader)
        dl_train, dl_val = make_loaders(cfg, transforms)
        logger.info("✅ DataLoaders ready "
                    f"— Train: {len(dl_train.dataset)} | Val: {len(dl_val.dataset)} | "
                    f"Batch size: {cfg['batch_size']}")

        # Keep first train loader for shallow autoencoder pretrain
        if first_dl_train_for_ae is None:
            first_dl_train_for_ae = dl_train




    # Pretrain shallow AE once (on first dataset’s images)
    pretrain_autoencoder(first_dl_train_for_ae, device, epochs=10, lr=1e-3, save_path=str(RESULTS_DIR / "shallow_encoder_pretrained.pth"))


    # Now actually train all three models sequentially for each dataset size
    # Train 3 models on all dataset sizes
    for n_cls in CLASS_SIZES:
        cfg["num_classes"] = n_cls
        dl_train, dl_val = make_loaders(cfg, transforms)



        # 1) Shallow CNN (uses pretrained encoder)
        print("\n==============================", file=results_log)
        print(f"Training Shallow CNN (Pretrained) | {n_cls} classes", file=results_log)
        print("==============================", file=results_log)
        logger.info(f"\nTraining Shallow CNN (Pretrained) | {n_cls} classes")
        shallow = ShallowCNN(n_cls, pretrained_path=str(RESULTS_DIR / "shallow_encoder_pretrained.pth")).to(device)
        train_model(shallow, cfg, dl_train, dl_val, device, model_tag="shallowcnn", classes_tag=str(n_cls))


        # 2) Deep CNN (ResNet backbone)
        print("\n==============================", file=results_log)
        print(f"Training Deep CNN (ResNet Pretrained) | {n_cls} classes", file=results_log)
        print("==============================", file=results_log)
        logger.info(f"\nTraining Deep CNN (ResNet Pretrained) | {n_cls} classes")
        deep = DeepCNNPretrained(n_cls, pretrained=True).to(device)
        train_model(deep, cfg, dl_train, dl_val, device, model_tag="deepcnn", classes_tag=str(n_cls))



        # 3) Vision Transformer (ViT)
        print("\n==============================", file=results_log)
        print(f"Training Vision Transformer (ViT-B16 Pretrained) | {n_cls} classes", file=results_log)
        print("==============================", file=results_log)
        logger.info(f"\nTraining Vision Transformer (ViT-B16 Pretrained) | {n_cls} classes")
        vit = ViTModel(n_cls, pretrained=True).to(device)
        train_model(vit, cfg, dl_train, dl_val, device, model_tag="vitb16", classes_tag=str(n_cls))

    results_log.write("All experiments (50 → 75 → 100) completed successfully!\n")
    results_log.close()
    logger.info("All experiments (50 → 75 → 100) completed successfully!")
