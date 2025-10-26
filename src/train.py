# src/train.py
import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from src.utils.seed import set_seed
from src.transforms import build_transforms
from src.data import make_loaders
from src.models import CNNShallow, CNNDeep, ViTTiny

def build_model(cfg, num_classes):
    if cfg.model.name == "cnn_shallow": return CNNShallow(num_classes)
    if cfg.model.name == "cnn_deep":    return CNNDeep(num_classes)
    if cfg.model.name == "vit_tiny":    return ViTTiny(num_classes, pretrained=cfg.model.pretrained)
    raise ValueError("unknown model")

def train(cfg):
    set_seed(cfg.seed)
    tfm = build_transforms(cfg.img_size)
    dl_train, dl_val = make_loaders(cfg, tfm)
    num_classes = max(max(y for _,y in dl_train.dataset), max(y for _,y in dl_val.dataset)) + 1

    model = build_model(cfg, num_classes).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(cfg.epochs):
        model.train(); total=0; correct=0; loss_sum=0.0
        for x,y in dl_train:
            x,y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                logits = model(x)
                loss = loss_fn(logits,y)
            scaler.scale(loss).then_backward()
            scaler.step(opt); scaler.update()
            loss_sum += loss.item()*x.size(0)
            pred = logits.argmax(1); correct += (pred==y).sum().item(); total += x.size(0)
        sched.step()
        train_acc = correct/total; train_loss = loss_sum/total

        # validation
        model.eval(); vtotal=0; vcorrect=0
        with torch.no_grad():
            for x,y in dl_val:
                x,y = x.to(cfg.device), y.to(cfg.device)
                logits = model(x); vcorrect += (logits.argmax(1)==y).sum().item(); vtotal += x.size(0)
        val_acc = vcorrect/vtotal
        print(f"epoch {epoch+1}/{cfg.epochs} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        if val_acc > best:
            best = val_acc
            torch.save({"model":model.state_dict(),"cfg":cfg.__dict__}, f"best_{cfg.model.name}.pt")

if __name__ == "__main__":
    from src.utils.config import load_cfg
    cfg = load_cfg()  # merges base.yaml + CLI overrides
    train(cfg)
