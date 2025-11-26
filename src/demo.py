"""
demo.py — Simple demonstration of how the Google Landmarks Recognition
pipeline works using the modules in this project.

This script:
1. Loads a small batch from the dataset
2. Initializes a ShallowCNN model
3. Runs a forward pass
4. Prints sample inputs and outputs
"""

import torch
from src.data import make_loaders
from src.transforms import get_transforms
from src.models import ShallowCNN

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

cfg = {
    "split_dir": "data/splits_balanced",
    "img_root": "data/images", #Refactor the path where images are loaded
    "batch_size": 4,        # small batch for demo
    "num_workers": 2,
    "num_classes": 50,      # use 50-class demo split
}

transforms = get_transforms(img_size=224)

# ---------------------------------------------------------
# 2. LOAD ONE BATCH
# ---------------------------------------------------------

dl_train, dl_val = make_loaders(cfg, transforms)

sample_batch = next(iter(dl_train))
images, labels = sample_batch

print("\n===== SAMPLE INPUT BATCH =====")
print("Image batch shape:", images.shape)     # Expected: [4, 3, 224, 224]
print("Labels:", labels.tolist())

# ---------------------------------------------------------
# 3. INITIALIZE MODEL
# ---------------------------------------------------------

model = ShallowCNN(num_classes=cfg["num_classes"])
model.eval()  # no training, just testing forward pass

# ---------------------------------------------------------
# 4. FORWARD PASS
# ---------------------------------------------------------

with torch.no_grad():
    outputs = model(images)

print("\n===== MODEL OUTPUT =====")
print("Logits shape:", outputs.shape)    # Expected: [4, num_classes]
print("Sample logits row 0:", outputs[0])

# ---------------------------------------------------------
# 5. PREDICTIONS
# ---------------------------------------------------------

preds = outputs.argmax(dim=1)
print("\n===== PREDICTIONS =====")
print("Predicted classes:", preds.tolist())

print("\nDemo complete!")
