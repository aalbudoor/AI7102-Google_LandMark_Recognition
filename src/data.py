import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
import logging

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# ID Normalization — Always Keep .jpg
# ============================================================
def _safe_id(x) -> str:
    """Normalize image id from CSV: strip spaces/quotes and lowercase (ensure .jpg extension)."""
    s = str(x).strip().strip('"').strip("'").lower()
    if not s.endswith(".jpg"):
        s += ".jpg"
    return s


# ============================================================
# Dataset Class for train/test images
# ============================================================
class GLDv2Dataset(Dataset):
    """
    Loads images from a flat directory (train_images or test_images) referenced by CSV.
    CSV column 'id' must match filenames (case-insensitive).
    """

    def __init__(self, csv_path, img_root, transform=None):
        self.csv_path = Path(csv_path)
        self.img_root = Path(img_root).resolve()
        self.transform = transform

        logger.info(f"Initializing dataset from CSV: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.df["id"] = self.df["id"].apply(_safe_id)

        logger.info(f"✅ Total CSV samples: {len(self.df)}")
        logger.info(f"🖼️ Image directory: {self.img_root}")

        # Map lowercase filename → full path
        self.image_map = self._index_images(self.img_root)

        found = sum(1 for img_id in self.df["id"] if img_id in self.image_map)
        logger.info(f"📊 Found {found:,}/{len(self.df):,} images ({found/len(self.df)*100:.2f}%) in directory.")

        sample_ids = list(self.df["id"].head(3))
        sample_hits = [sid in self.image_map for sid in sample_ids]
        logger.info(f"🔍 Sample check (first 3): {list(zip(sample_ids, sample_hits))}")

    def _index_images(self, root_dir: Path):
        """Builds map: <lowercase filename> → full path."""
        if not root_dir.exists():
            logger.error(f"❌ Image directory not found: {root_dir}")
            return {}

        image_map = {}
        valid_exts = {".jpg", ".jpeg", ".png"}
        total = 0

        for p in root_dir.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                image_map[p.name.lower()] = p
                total += 1

        logger.info(f"📸 Indexed {total:,} image files from {root_dir}")
        return image_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"].lower()
        label = int(row["label"])

        img_path = self.image_map.get(img_id)
        if not img_path or not img_path.exists():
            logger.warning(f"⚠️ Missing image: {img_id} not found. Using black placeholder.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"⚠️ Error loading {img_id} ({e}). Using black placeholder.")
                image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# DataLoader Builder
# ============================================================
def make_loaders(cfg, transforms):
    split_dir = Path(cfg.get("split_dir", "data/splits_balanced")).resolve()
    num_classes = cfg.get("num_classes", 100)

    # ✅ Updated paths for new dataset locations
    train_csv = split_dir / f"train_{num_classes}.csv"
    test_csv = split_dir / f"test_{num_classes}.csv"
    train_img_root = Path(cfg.get("train_img_root", "data/train_images")).resolve()
    test_img_root = Path(cfg.get("test_img_root", "data/test_images")).resolve()

    logger.info(f"📁 Loading train/test CSVs from: {split_dir}")
    logger.info(f"📦 Train images root: {train_img_root}")
    logger.info(f"📦 Test images root: {test_img_root}")

    train_ds = GLDv2Dataset(train_csv, train_img_root, transform=transforms["train"])
    val_ds = GLDv2Dataset(test_csv, test_img_root, transform=transforms["val"])

    logger.info(f"✅ Datasets ready — Train: {len(train_ds)} | Val: {len(val_ds)}")

    dl_train = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )

    dl_val = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )

    return dl_train, dl_val


# ============================================================
# Default Transforms
# ============================================================
default_transforms = {
    "train": T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]),
    "val": T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ]),
}
