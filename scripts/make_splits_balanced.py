# scripts/make_splits_balanced.py
from pathlib import Path
import sys, json
import pandas as pd

# ---------- paths (match your current layout) ----------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT    = PROJECT_ROOT / "data"
RAW_ROOT     = DATA_ROOT / "raw"
IMG_ROOT     = RAW_ROOT / "train"                 # GLDv2 fan-out lives here
META_DIR     = RAW_ROOT / "metadata"
META         = META_DIR / "train.csv"
OUT          = DATA_ROOT / "splits"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- knobs ----------
TARGET_CLASSES        = 200      # 60–100 is typical; adjust as you like
MIN_PER_CLASS_START   = 20      # relaxes to 15/10/8/5/3/2 automatically
MIN_SAMPLES_CLASS     = 12      # drop classes with <12 total
CAP_PER_CLASS         = 50      # cap total/class BEFORE split

TEST_SIZE             = 0.20
MIN_TRAIN_PER_CLASS   = 8       # enforce per-split minima
MIN_VAL_PER_CLASS     = 3
RANDOM_STATE          = 42

print("META     :", META)
print("IMG_ROOT :", IMG_ROOT)
print("OUT      :", OUT)

# ---------- sanity ----------
if not META.exists(): sys.exit(f"❌ Missing {META}")
if not IMG_ROOT.exists(): sys.exit(f"❌ Missing images folder {IMG_ROOT}")

ids_on_disk = {p.stem for p in IMG_ROOT.rglob("*.jpg")}
if not ids_on_disk:
    sys.exit("❌ No .jpg files found under IMG_ROOT. Did you extract the tars?")

df = pd.read_csv(META, usecols=["id","landmark_id"], dtype={"id": str, "landmark_id": int})
df = df[df["id"].isin(ids_on_disk)].copy()
if df.empty:
    sys.exit("❌ No metadata rows matched files on disk (check fan-out path a/b/c/id.jpg).")

# ---------- select classes: relax + fill ----------
vc_all = df["landmark_id"].value_counts()
selected = []
for m in [MIN_PER_CLASS_START, 15, 10, 8, 5, 3, 2]:
    cand = vc_all[vc_all >= m].index.tolist()
    if len(cand) >= 2:
        selected = cand[:TARGET_CLASSES]  # take most populated first
        print(f"Using MIN_PER_CLASS={m}: found {len(cand)} classes (taking {len(selected)})")
        break

if len(selected) < TARGET_CLASSES:
    filler = vc_all[~vc_all.index.isin(selected)]
    filler = filler[filler >= 2].index.tolist()
    need = TARGET_CLASSES - len(selected)
    take = filler[:need]
    selected += take
    print(f"Filled with {len(take)} more classes (≥2 imgs) -> total {len(selected)}")

if len(selected) < 2:
    sys.exit("❌ Need at least 2 classes. Lower TARGET_CLASSES/thresholds or add data.")

# ---------- filter, drop tiny, cap heads ----------
sub = df[df["landmark_id"].isin(selected)].copy()

if MIN_SAMPLES_CLASS and MIN_SAMPLES_CLASS > 1:
    counts = sub["landmark_id"].value_counts()
    keep_labels = counts[counts >= MIN_SAMPLES_CLASS].index
    dropped = set(sub["landmark_id"].unique()) - set(keep_labels)
    if dropped:
        sub = sub[sub["landmark_id"].isin(keep_labels)].copy()
        print(f"Dropped {len(dropped)} classes with <{MIN_SAMPLES_CLASS} total "
              f"→ remaining classes: {sub['landmark_id'].nunique()}")

# if still more than target, keep top-K by frequency
if sub["landmark_id"].nunique() > TARGET_CLASSES:
    top_labels = sub["landmark_id"].value_counts().head(TARGET_CLASSES).index
    sub = sub[sub["landmark_id"].isin(top_labels)].copy()
    print(f"Trimmed to top-{TARGET_CLASSES} classes by frequency.")

# map labels now
labels_sorted = sorted(sub["landmark_id"].unique().tolist())
label2idx = {int(l): i for i, l in enumerate(labels_sorted)}
idx2label = {v: k for k, v in label2idx.items()}
sub["y"] = sub["landmark_id"].map(label2idx).astype(int)

# cap per class BEFORE splitting (fix pandas future warning with include_groups=False)
if CAP_PER_CLASS:
    sub = (
        sub.groupby("landmark_id", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), CAP_PER_CLASS), random_state=RANDOM_STATE),
                  include_groups=False)
           .reset_index(drop=True)
    )
    print(f"Capped each class to ≤{CAP_PER_CLASS}. Total after cap: {len(sub)}")

# ---------- per-class split enforcing minima ----------
required_total = max(MIN_SAMPLES_CLASS, MIN_TRAIN_PER_CLASS + MIN_VAL_PER_CLASS)

def per_class_split(df_cls, test_size, min_train, min_val, seed):
    n = len(df_cls)
    val_n = max(min_val, int(round(test_size * n)))
    train_n = n - val_n
    if train_n < min_train:
        need = min_train - train_n
        if val_n - need >= min_val:
            val_n -= need
            train_n = n - val_n
    if train_n < min_train or val_n < min_val:
        return None, None
    df_shuf = df_cls.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df  = df_shuf.iloc[:val_n]
    train_df= df_shuf.iloc[val_n:]
    return train_df, val_df

train_parts, val_parts, dropped_y = [], [], []
for y_val, g in sub.groupby("y"):
    tr, va = per_class_split(
        g[["id","y"]],
        test_size=TEST_SIZE,
        min_train=MIN_TRAIN_PER_CLASS,
        min_val=MIN_VAL_PER_CLASS,
        seed=RANDOM_STATE + int(y_val),
    )
    if tr is None:
        dropped_y.append(int(y_val))
        continue
    train_parts.append(tr); val_parts.append(va)

if dropped_y:
    print(f"Dropped {len(dropped_y)} classes that couldn’t satisfy ≥{MIN_TRAIN_PER_CLASS} train & ≥{MIN_VAL_PER_CLASS} val.")

if not train_parts or not val_parts:
    sys.exit("❌ No classes left after enforcing minima. Lower TARGET_CLASSES/thresholds and retry.")

train_df = pd.concat(train_parts, ignore_index=True)
val_df   = pd.concat(val_parts,   ignore_index=True)

# remap to contiguous indices AFTER drops
kept_y = sorted(set(train_df["y"]).union(set(val_df["y"])))
final_labels = [idx2label[y] for y in kept_y]           # original landmark_ids
final_label2idx = {int(l): i for i, l in enumerate(final_labels)}
remap = {y:i for i, y in enumerate(kept_y)}
train_df["y"] = train_df["y"].map(remap).astype(int)
val_df["y"]   = val_df["y"].map(remap).astype(int)

# verify minima
tr_min = int(train_df["y"].value_counts().min())
va_min = int(val_df["y"].value_counts().min())
if tr_min < MIN_TRAIN_PER_CLASS or va_min < MIN_VAL_PER_CLASS:
    sys.exit(f"❌ Split failed minima after per-class split "
             f"(train_min={tr_min} wanted ≥{MIN_TRAIN_PER_CLASS}, "
             f"val_min={va_min} wanted ≥{MIN_VAL_PER_CLASS}). "
             f"Lower TARGET_CLASSES/thresholds and retry.")

# ---------- write ----------
train_df.to_csv(OUT / "train.csv", index=False)
val_df.to_csv(OUT / "val.csv", index=False)
with open(OUT / "labelmap.json", "w") as f:
    json.dump(final_label2idx, f)

print(f"✅ Done. Classes: {len(final_label2idx)} | train: {len(train_df)} | val: {len(val_df)}")
print(f"Per-class mins: train ≥{tr_min}, val ≥{va_min}")
