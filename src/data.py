# src/data.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

class GLDv2Dataset(Dataset):
    """
    A dataset loader for merged GLDv2 CSVs that contain full file paths.
    Expected CSV columns: id, y (label), filepath, ...
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Basic sanity check
        if "filepath" not in self.df.columns:
            raise ValueError(f"'filepath' column missing in {csv_path}. "
                             "Make sure you merged splits with metadata.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row.filepath)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")

        if self.transform:
            img = self.transform(img)
        label = int(row.label)
        return img, label


def make_loaders(cfg, transforms):
    """
    Create train and validation DataLoaders using merged_full split CSVs.
    """
    base_dir = Path("data/splits_80_10_10/merged_full")  # or 70_15_15, configurable
    train_csv = base_dir / "train_full.csv"
    val_csv   = base_dir / "val_full.csv"

    train_ds = GLDv2Dataset(train_csv, transform=transforms["train"])
    val_ds   = GLDv2Dataset(val_csv, transform=transforms["val"])

    dl_train = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    return dl_train, dl_val
