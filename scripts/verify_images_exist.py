from pathlib import Path
import pandas as pd

# --- Configuration ---
splits = [
    "data/splits_70_15_15/merged_full/train_full.csv",
    "data/splits_70_15_15/merged_full/val_full.csv",
    "data/splits_70_15_15/merged_full/test_full.csv",
]

# Root directory where all your shards and nested images are stored
img_root = Path("/Users/abdullaalbudoor/Desktop/raw/train_images_shards")

# --- Step 1: build a fast lookup of all available image IDs ---
print("🔍 Indexing all .jpg files under:", img_root)
all_jpgs = list(img_root.rglob("*.jpg"))  # recursively search all subfolders
print(f"Found {len(all_jpgs):,} image files total.")

# Create a set of image IDs (filenames without .jpg)
available_ids = {p.stem for p in all_jpgs}
print("Example image IDs found:", list(available_ids)[:5])

# --- Step 2: verify each CSV split ---
for split_path in splits:
    df = pd.read_csv(split_path)
    total = len(df)
    ids = set(df["id"])
    missing = ids - available_ids  # set difference
    extra = available_ids - ids    # optional: images not listed

    print(f"\n📁 Checking split: {split_path}")
    print(f"  Total entries in CSV: {total}")
    print(f"  Missing images: {len(missing)}")
    print(f"  Extra images not in CSV (optional): {len(extra)}")

    if missing:
        print("  Example missing IDs:", list(missing)[:5])
