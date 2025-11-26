import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
import logging

# Logging Configuration — makes logs readable and timestamped
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ID Normalization Helper — Always returns a .jpg filename
def _safe_id(x) -> str:
    """
    Clean up an ID string coming from the CSV file.

    Operations:
    - Convert to string
    - Strip whitespace + surrounding quotes
    - Lowercase everything (file lookup becomes case-insensitive)
    - Ensure the filename ends with .jpg

    This is because GLDv2 CSVs sometimes contain IDs like 'AbC123' or "abc123.JPG".
    This normalizes everything so it matches the actual filenames on disk.
    """
    s = str(x).strip().strip('"').strip("'").lower()
    if not s.endswith(".jpg"):
        s += ".jpg"
    return s


# GLDv2 Dataset Class, Loads images referenced by a CSV file
class GLDv2Dataset(Dataset):
    """
    Expected:
    - A CSV file (train/test split) with columns: id, label
    - A directory containing all images (train_images or test_images)
    - The 'id' column must correspond to filenames in img_root

    Responsibilities of this class:
    - Read CSV
    - Normalize all IDs using _safe_id
    - Build a lookup table mapping "filename.jpg" → full file path
    - Load images from disk, with fallback to a black image if missing
    """

    def __init__(self, csv_path, img_root, transform=None):
        self.csv_path = Path(csv_path)
        self.img_root = Path(img_root).resolve()
        self.transform = transform

        logger.info(f"Initializing dataset from CSV: {self.csv_path}")

        # Load CSV and normalize image IDs        
        self.df = pd.read_csv(self.csv_path)
        self.df["id"] = self.df["id"].apply(_safe_id)

        logger.info(f"Total CSV samples: {len(self.df)}")
        logger.info(f"Image directory: {self.img_root}")

        # Build a mapping of {lowercase filename → full file path}
        self.image_map = self._index_images(self.img_root)

        # Count how many images actually exist on disk
        found = sum(1 for img_id in self.df["id"] if img_id in self.image_map)
        logger.info(f"📊 Found {found:,}/{len(self.df):,} images ({found/len(self.df)*100:.2f}%) in directory.")

        sample_ids = list(self.df["id"].head(3))
        sample_hits = [sid in self.image_map for sid in sample_ids]
        logger.info(f"🔍 Sample check (first 3): {list(zip(sample_ids, sample_hits))}")



    def _index_images(self, root_dir: Path):
        """
        Scan the image directory and create a dictionary:
        {
            "filename.jpg": <Path object to file>
        }

        This is because file systems differ (Linux = case-sensitive, Windows = not).
        Lowercasing everything avoids mismatches like "abc.jpg" vs "ABC.JPG".
        """
        if not root_dir.exists():
            logger.error(f"Image directory not found: {root_dir}")
            return {}

        image_map = {}
        valid_exts = {".jpg", ".jpeg", ".png"}
        total = 0

        # Iterate through all files in the directory
        for p in root_dir.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                image_map[p.name.lower()] = p
                total += 1

        logger.info(f"Indexed {total:,} image files from {root_dir}")
        return image_map

    # This function retruns of rows in the CSV (one per sample).
    def __len__(self):
        return len(self.df)

    # This function fetch one sample, look up image ID, load the image using PIL, apply transformations (if any), then lastly return (image_tensor, label)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"].lower()
        label = int(row["label"])

        # Locate image file
        img_path = self.image_map.get(img_id)

        # Missing image fallback: return a black 224×224 image
        if not img_path or not img_path.exists():
            logger.warning(f"Missing image: {img_id} not found. Using black placeholder.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        else:
            try:
                # .convert("RGB") ensures consistent 3-channel input                
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Error loading {img_id} ({e}). Using black placeholder.")
                image = Image.new("RGB", (224, 224), (0, 0, 0))


        # Apply transforms (if provided)
        if self.transform:
            image = self.transform(image)
        return image, label





# DataLoader Builder — returns (train_loader, val_loader)
def make_loaders(cfg, transforms):
    """
    Creates PyTorch DataLoaders for training/validation.

    cfg fields used:
    - split_dir: folder containing train_X.csv / test_X.csv
    - num_classes: determines which CSVs to load
    - batch_size
    - num_workers
    - train_img_root / test_img_root (optional)
    """
    split_dir = Path(cfg.get("split_dir", "data/splits_balanced")).resolve()
    num_classes = cfg.get("num_classes", 100)

    # CSV files for this experiment
    train_csv = split_dir / f"train_{num_classes}.csv"
    test_csv = split_dir / f"test_{num_classes}.csv"

    # Image directories
    train_img_root = Path(cfg.get("train_img_root", "data/train_images")).resolve()
    test_img_root = Path(cfg.get("test_img_root", "data/test_images")).resolve()

    logger.info(f"Loading train/test CSVs from: {split_dir}")
    logger.info(f"Train images root: {train_img_root}")
    logger.info(f"Test images root: {test_img_root}")

    # Create dataset objects
    train_ds = GLDv2Dataset(train_csv, train_img_root, transform=transforms["train"])
    val_ds = GLDv2Dataset(test_csv, test_img_root, transform=transforms["val"])

    logger.info(f"Datasets ready — Train: {len(train_ds)} | Val: {len(val_ds)}")


    # Build PyTorch DataLoaders
    dl_train = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4), pin_memory=True, persistent_workers=cfg.get("num_workers", 4) > 0,)

    dl_val = DataLoader(val_ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg.get("num_workers", 4),pin_memory=True,persistent_workers=cfg.get("num_workers", 4) > 0,)

    return dl_train, dl_val


# Default Transforms — fallbacks if user does not provide their own
default_transforms = {
    "train": T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(), # data augmentation
        T.ToTensor(),
    ]),
    "val": T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ]),
}
