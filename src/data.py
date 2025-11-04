# src/data.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class GLDv2Dataset(Dataset):
    """
    Dataset for your custom 100-class Google Landmarks subset.
    Expects columns: id, label, (optional: url, landmark_id, image_path)
    """
    def __init__(self, csv_path, img_root, transform=None):
        start = time.time()
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.transform = transform

        logger.info(f"Initializing GLDv2Dataset from {csv_path}")
        logger.info(f"Total samples: {len(self.df)}")
        logger.info(f"Image root: {self.img_root}")

        # Preload all image paths once for faster access
        logger.info("Indexing all .jpg image files... (first time may take ~10–30s)")
        self.all_images = {p.stem: p for p in self.img_root.rglob("*.jpg")}
        logger.info(f"Indexed {len(self.all_images):,} images in {time.time() - start:.2f}s")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row.id)

        img_path = self.all_images.get(img_id)
        if img_path is None:
            logger.warning(f"⚠️ Missing image: {img_id}.jpg not found under {self.img_root}")
            raise FileNotFoundError(f"Image {img_id}.jpg not found under {self.img_root}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"❌ Failed to open {img_id}.jpg: {e}")
            raise

        if self.transform:
            img = self.transform(img)

        label = int(row.label)
        return img, label

    def __len__(self):
        return len(self.df)


def make_loaders(cfg, transforms):
    """
    Creates DataLoaders for your 100-class dataset.
    """
    # === Update paths for your new dataset ===
    base_dir = Path("data/splits_100")  # You can store CSVs here or adjust path
    img_root = Path(cfg["img_root"])

    train_csv = base_dir / "train_100.csv"
    val_csv   = base_dir / "test_100.csv"

    logger.info(f"Loading datasets from: {base_dir}")

    # === Create Datasets ===
    train_ds = GLDv2Dataset(train_csv, img_root, transform=transforms["train"])
    val_ds   = GLDv2Dataset(val_csv, img_root, transform=transforms["val"])

    # === Create DataLoaders ===
    dl_train = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
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

    logger.info(f"✅ DataLoaders ready — Train: {len(train_ds)} | Val: {len(val_ds)}")
    return dl_train, dl_val
