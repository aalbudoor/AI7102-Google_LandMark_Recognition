#!/usr/bin/env python
from pathlib import Path
import pandas as pd

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
TRAIN_IMG    = RAW_DIR / "train"
TEST_IMG     = RAW_DIR / "test"     # create & extract test shards here if you have them
META_DIR     = RAW_DIR / "metadata"
OUT_DIR      = PROJECT_ROOT / "data" / "filtered_metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
POSSIBLE_ID_COLS = ["id", "image_id", "retrieval_image_id", "ImageID", "image"]
# If True, keep only public portion of recognition_solution_v2.1.csv
KEEP_PUBLIC_ONLY = True

# files that are NOT per-image (copy as-is)
PER_LABEL_FILES = {
    "train_label_to_category.csv",
}

# Map CSV -> which image root to filter against
CSV_TO_IMGROOT = {
    "train.csv": TRAIN_IMG,
    "train_attribution.csv": TRAIN_IMG,
    "train_clean.csv": TRAIN_IMG,   # many variants have no id column; handled below
    "test.csv": TEST_IMG,           # if TEST_IMG doesn't exist, we'll fallback to TRAIN_IMG to avoid crashes
    "recognition_solution_v2.1.csv": TEST_IMG,
}

def ids_on_disk(img_root: Path):
    if not img_root.exists():
        return set()
    return {p.stem for p in img_root.rglob("*.jpg")}

def detect_id_col(df: pd.DataFrame):
    for col in POSSIBLE_ID_COLS:
        if col in df.columns:
            return col
    return None

def filter_by_ids(df: pd.DataFrame, img_ids: set, id_col_hint: str | None = None):
    id_col = id_col_hint or detect_id_col(df)
    if id_col is None:
        return df, None
    df[id_col] = df[id_col].astype(str)
    df_f = df[df[id_col].isin(img_ids)].copy()
    return df_f, id_col

def process_csv(csv_name: str):
    src = META_DIR / csv_name
    if not src.exists():
        print(f"(skip) {src} not found")
        return

    # Per-label files: copy as-is
    if csv_name in PER_LABEL_FILES:
        df = pd.read_csv(src)
        dst = OUT_DIR / csv_name
        df.to_csv(dst, index=False)
        print(f"→ {csv_name}: (per-label) copied {len(df):,} rows -> {dst}")
        return

    # Load
    df = pd.read_csv(src)

    # Special handling for recognition_solution_v2.1.csv
    if csv_name == "recognition_solution_v2.1.csv":
        # Optionally keep only Usage == Public
        if KEEP_PUBLIC_ONLY and "Usage" in df.columns:
            before = len(df)
            df = df[df["Usage"] == "Public"].copy()
            print(f"→ {csv_name}: kept 'Public' rows {len(df):,}/{before:,}")

    # Decide which image root to filter against
    img_root = CSV_TO_IMGROOT.get(csv_name, TRAIN_IMG)
    if not img_root.exists():
        # if test root missing, fall back to train (user may have placed test images in train)
        if csv_name in ("test.csv", "recognition_solution_v2.1.csv"):
            print(f"⚠️  {img_root} missing; falling back to TRAIN_IMG={TRAIN_IMG} for {csv_name}")
            img_root = TRAIN_IMG

    img_ids = ids_on_disk(img_root)

    # Filter by IDs if the file has an image-id-like column
    df_f, id_col = filter_by_ids(df, img_ids)

    # Write
    dst = OUT_DIR / csv_name
    df_f.to_csv(dst, index=False)
    if id_col is None:
        print(f"→ {csv_name}: (no image-id column found) wrote {len(df_f):,} rows unfiltered -> {dst}")
    else:
        print(f"→ {csv_name}: kept {len(df_f):,} rows using id_col='{id_col}' -> {dst}")

def main():
    print("Train images root:", TRAIN_IMG)
    print(" Test images root:", TEST_IMG, "(exists)" if TEST_IMG.exists() else "(missing)")
    print("Metadata dir     :", META_DIR)
    # Process the common files you mentioned
    files = [
        "train.csv",
        "train_attribution.csv",
        "train_clean.csv",
        "train_label_to_category.csv",
        "test.csv",
        "recognition_solution_v2.1.csv",
    ]
    for name in files:
        process_csv(name)

if __name__ == "__main__":
    main()
