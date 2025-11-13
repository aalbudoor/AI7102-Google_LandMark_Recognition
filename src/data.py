from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import logging
import time

# Setup simple logging so we can see progress and debugging info in terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)



# This is for the Custom Dataset class for Google Landmark subsets (50, 75, 100 classes)
class GLDv2Dataset(Dataset):
    """
    This class tells PyTorch how to load and prepare each image and its label.
    It uses the CSV files (train_50.csv, test_50.csv, etc.) that we generated earlier
    """

    def __init__(self, csv_path, img_root, transform=None):
        start = time.time()

        # Load the CSV file which lists image IDs and their labels
        self.df = pd.read_csv(csv_path)

        # Path to the folder where all images are stored
        self.img_root = Path(img_root)

        # Any image transformations (like resize, normalize) to apply    
        self.transform = transform

        logger.info(f"Initializing GLDv2Dataset from {csv_path}")
        logger.info(f"Total samples: {len(self.df)}")
        logger.info(f"Image root: {self.img_root}")

        # Scan all folders and subfolders once to find every .jpg image
        # We store them in a dictionary like {"052bdf26...": "/path/to/image.jpg"}
        # so that we can quickly look up images later.
        logger.info("Indexing .jpg image files recursively:")
        self.all_images = {p.stem: p for p in self.img_root.rglob("*.jpg")}
        logger.info(f"Indexed {len(self.all_images):,} images in {time.time() - start:.2f}s")



    # This function is for how to load ONE sample (image + label) when the DataLoader asks for it
    def __getitem__(self, idx):
        row = self.df.iloc[idx] # Pick one row from the CSV
        img_id = str(row.id)

        # Find where the image file is located
        img_path = self.all_images.get(img_id)
        if img_path is None:
            logger.warning(f"Missing image: {img_id}.jpg not found under {self.img_root}")
            raise FileNotFoundError(f"Image {img_id}.jpg not found under {self.img_root}")

        # Try to open the image safely
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {img_id}.jpg: {e}")
            raise



        # Apply optional transformations (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        label = int(row.label)
        return img, label     # Return total number of samples in this dataset

    def __len__(self):
        return len(self.df)





def make_loaders(cfg, transforms):
    """
    This function creates the train and validation DataLoaders. 
    It automatically picks the correct CSV files (train_50.csv, test_50.csv, etc.) depending on how many classes you set in your config (50, 75, or 100).
    """

    # Find where the balanced CSVs are stored (defaults to data/splits_balanced)
    split_dir = Path(cfg.get("split_dir", "data/splits_balanced"))  # Default: new dataset folder
    num_classes = cfg.get("num_classes", 100)  # 50, 75, or 100
    img_root = Path(cfg["img_root"])

    # Choose which CSV files to use based on the number of classes
    train_csv = split_dir / f"train_{num_classes}.csv"
    val_csv = split_dir / f"test_{num_classes}.csv"

    # Sanity check — stop if we can’t find those CSVs
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Could not find train/test CSVs for {num_classes} classes under {split_dir}.\n"
            f"Expected: {train_csv.name}, {val_csv.name}"
        )

    logger.info(f"Loading datasets from: {split_dir}")
    logger.info(f"Number of classes: {num_classes}")






    # Create Dataset objects (one for training, one for validation)
    train_ds = GLDv2Dataset(train_csv, img_root, transform=transforms["train"])
    val_ds = GLDv2Dataset(val_csv, img_root, transform=transforms["val"])


    # Create DataLoaders — these actually feed the data to the model in batches

    dl_train = DataLoader(train_ds, batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["num_workers"],pin_memory=True)

    dl_val = DataLoader(val_ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["num_workers"],pin_memory=True)

    logger.info(f"DataLoaders ready — Train: {len(train_ds)} | Val: {len(val_ds)}")
    return dl_train, dl_val
