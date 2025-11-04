import pandas as pd

# Read the CSV file
df = pd.read_csv("merged_full/train_full.csv")

# Check that the 'label' column exists
if 'label' not in df.columns:
    raise ValueError("The CSV file must contain a 'label' column")

# Count occurrences of each label
label_counts = df['label'].value_counts()

# Print the results (descending order by default)
print("Label counts (descending):\n")
for label, count in label_counts.items():
    print(f"{label}: {count}")
