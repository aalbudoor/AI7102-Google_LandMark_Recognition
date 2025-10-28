# src/data.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

class GLDv2Dataset(Dataset):
    """
    Dataset for Google Landmarks subset using merged_full CSVs.
    Expects columns: id, label, (optional: filepath, url, landmark_id)
    """
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row.id

        # Try direct path first
        direct_path = self.img_root / f"{img_id}.jpg"
        if direct_path.exists():
            img_path = direct_path
        else:
            # 🔍 Recursively search for the file in subdirectories
            found = list(self.img_root.rglob(f"{img_id}.jpg"))
            if len(found) == 0:
                raise FileNotFoundError(f"Image {img_id}.jpg not found under {self.img_root}")
            img_path = found[0]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row.label)
        return img, label



    def __len__(self):
        return len(self.df)


def make_loaders(cfg, transforms):
    """
    Creates DataLoaders for train and validation splits.
    """
    base_dir = Path("data/splits_80_10_10/merged_full") # or 70_15_15, configurable
    img_root = cfg["img_root"]

    train_csv = base_dir / "train_full.csv"
    val_csv   = base_dir / "val_full.csv"

    train_ds = GLDv2Dataset(train_csv, img_root, transform=transforms["train"])
    val_ds   = GLDv2Dataset(val_csv, img_root, transform=transforms["val"])

    dl_train = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,       # simple random shuffle
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    dl_val = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    return dl_train, dl_val