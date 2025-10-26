import pandas as pd

# Load split file (e.g., 80_10_10 train)
#split_df = pd.read_csv("data/splits_80_10_10/train.csv")
split_df = pd.read_csv("data/splits_70_15_15/train.csv")

# Load filtered metadata (which contains full info)
meta_df = pd.read_csv("data/filtered_metadata/train.csv")

# Merge to restore metadata
merged = split_df.merge(meta_df, on="id", how="left")

print(len(split_df), "rows in split file")
print(len(merged.dropna()), "rows successfully matched in metadata")
print("Missing metadata rows:", len(split_df) - len(merged.dropna()))
