import pandas as pd
from pathlib import Path

# --- Paths ---
split_root = Path("data/splits_70_15_15")          # or data/splits_70_15_15
meta_path = Path("data/filtered_metadata/train.csv")  # filtered metadata with full info
output_dir = split_root / "merged_full"             # new folder for merged files
output_dir.mkdir(exist_ok=True)

# --- Load metadata ---
meta_df = pd.read_csv(meta_path)
print(f"Loaded metadata: {len(meta_df)} rows")

# Ensure consistent column names
if "landmark_id" in meta_df.columns and "y" not in meta_df.columns:
    meta_df = meta_df.rename(columns={"landmark_id": "y"})

# --- Helper function to merge each split ---
# Inside merge_splits_with_metadata.py
def merge_split(split_name):
    split_file = split_root / f"{split_name}.csv"
    df = pd.read_csv(split_file)
    merged = df.merge(meta_df, on="id", how="left")

    # ✅ Rename columns for clarity
    if "y_x" in merged.columns and "y_y" in merged.columns:
        merged = merged.rename(columns={"y_x": "label", "y_y": "landmark_id"})

    out_path = output_dir / f"{split_name}_full.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved merged file: {out_path}")


# --- Merge train/val/test ---
for name in ["train", "val", "test"]:
    merge_split(name)
