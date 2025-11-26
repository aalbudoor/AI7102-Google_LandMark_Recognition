# AI7102-Google_LandMark_Recognition
Automatic landmark recognition is a challenging problem: wide class imbalance, intra-class variation (lighting, viewpoint, occlusion), and inter-class similarity. The proposed solutions use convolutional architecture, metric learning, retrieval based methods, and data augmentation.

This project implements a full training pipeline for the **Google Landmark Recognition Dataset (GLDv2)** using:

- **Custom Shallow CNN**
- **Deep CNN with ResNet18 backbone**
- **Vision Transformer (ViT-B16)**
- **Variable-depth CNN experiments (3–13 layers)**
- **Autoencoder pretraining for feature initialization**

Purpose is to evaluates model performance across **50-class**, **75-class**, and **100-class** subsets of GLDv2, and includes both **CNN depth experiments** and **full-model experiments**.

---

## **Project Directory Structure**



```
AI7102-Google_LandMark_Recognition/
│
├── data/
│   ├── images/                  # Downloaded GLDv2 images
│   ├── filtered_images/         # (Optional) Cleaned / curated images
│   ├── splits_balanced/         # CSV splits: train_50.csv, test_50.csv ...
│   ├── train_images/            # Original GLDv2 training images
│   ├── test_images/             # Original GLDv2 test images
│   ├── *.csv                    # Metadata + label mappings
│
├── scripts/
│   ├── download-dataset.sh      # Downloads GLDv2 images & metadata
│   ├── flatten_images.sh        # Utility: flattens directory structure
│   ├── Visual_class_Distribution
│
├── cnn_depth_experiments/       # Contains CNN depth experiment results
│   └── <timestamp>/
│       └── training_summary.txt
│
├── runs/                        # All runs from train_all_models.py
│   └── <timestamp>/
│       └── training_summary.txt
│
├── src/
│   ├── data.py                  # Dataset loader (GLDv2 Dataset)
│   ├── models.py                # Shallow CNN, Deep CNN, ViT
│   ├── transforms.py            # Augmentation + normalization
│   ├── utils/                   # (Optional utilities)
│   ├── train_all_models.py      # Main experiment: shallow, deep, ViT
│   ├── train_all_cnns.py        # CNN depth experiment (3–13 layers)
|   |-- demo.py                  # Pipeline Demonstration Script
│
├── README.md                    # ← Place this file here
├── requirements.txt             # ← Also place this beside README
```


This project uses **Google Landmark Dataset v2** (GLDv2).  
The dataset is downloaded using:

/scripts/download-dataset.sh

This script follows the instructions from the original project:

**Official dataset instructions:**  
https://github.com/cvdfoundation/google-landmark

Using the provided script in /scripts/download-dataset.sh

Run the command (Make sure permissions are enabled using "chmod +x download-dataset.sh")

```mkdir train && cd train```
```bash ../download-dataset.sh train 499```

This will automatically download, verify and extract the images to the train directory.



The downloaded images will be placed inside:


data/train_images/
data/test_images/


** CSV metadata (train_50.csv, test_50.csv, etc.) is stored in: **

data/splits_balanced/


---

# **2. Training Pipelines**

## **A. train_all_models.py**  
This is the **main multi-model experiment script**.

### **What it does:**
- Loads datasets for **50, 75, and 100 classes**  
-  Pretrains a **Shallow Autoencoder** once  
- Then trains the following models on each dataset size:

| Model | Description |
|-------|-------------|
| **ShallowCNN** | Tiny CNN initialized using autoencoder encoder weights |
| **DeepCNNPretrained** | ResNet18 backbone + extra conv blocks |
| **ViTModel** | Pretrained Vision Transformer from TIMM |

### **Pipeline Steps**
1. Load train/validation splits  
2. Pretrain Autoencoder → saves encoder weights  
3. Train ShallowCNN(50), then DeepCNN(50), then ViT(50)  
4. Repeat for 75 and 100 classes  
5. Save the **best checkpoint** per model and dataset  
6. Log results to `runs/<timestamp>/training_summary.txt`

---

## **B. train_all_cnns.py**  
A dedicated experiment to investigate **network depth** in simple CNNs.

### **Depth values used:**  
[3, 5, 7, 9, 11, 13]



### **Pipeline Steps**
1. Load dataset (default: 100 classes)  
2. Pretrain a shallow autoencoder  
3. Build CNN with increasing depth:
   - Conv2D → ReLU → MaxPool, repeated *depth* times  
4. Train each CNN from scratch  
5. Save best checkpoint for every depth in:  cnn_depth_experiments/<timestamp>/



This lets you analyze how CNN depth affects accuracy.

---

# **3. src/ — Core Modules**

## **src/data.py**
Implements:

### **GLDv2Dataset**
- Loads images referenced in CSV files  
- Normalizes IDs like `"Abc123.JPG"` → `"abc123.jpg"`  
- Handles missing images gracefully  
- Applies training/validation transforms  

### **make_loaders()**
Returns: (train_loader, val_loader)

with:

- balanced splits based on train_X.csv / test_X.csv  
- selected image root directory  
- selected batch size  
- pinned memory  
- persistent workers  

---

## **src/models.py**  
Contains all model architectures.

### **ShallowAutoencoder**
- Used ONLY for pretraining  
- encoder: Conv → ReLU → MaxPool  
- decoder: ConvTranspose → Sigmoid  

### **ShallowCNN**
- classifier using the autoencoder encoder  
- supports pretrained encoder loading  

### **DeepCNNPretrained**
- ResNet18 backbone  
- additional conv blocks  
- classifier head (256 → num_classes)  

### **ViTModel**
- ViT-Base Patch-16  
- loaded via **timm**  
- supports pretrained ImageNet weights  

---

## **src/transforms.py**
Defines all transformations.

### Training transforms:
- RandomResizedCrop  
- Flip  
- Rotation  
- Color jitter  
- Affine distortions  
- Normalize (ImageNet mean/std)  

### Validation transforms:
- Resize  
- Normalize  

---

# **4. Running Experiments**

## **Train All Models**
- Ensure you are in the root directory and run it using the following:
```python -m src.train_all_models ```


Output saved in: runs/<timestamp>/


## **Run CNN Depth Experiments**
- Ensure you are in the root directory and run it using the following:
```python -m src.train_all_cnns ```


Output saved in: cnn_depth_experiments/<timestamp>/


# How to Use the `requirements.txt` File

This project includes a `requirements.txt` file that lists all Python dependencies needed to run the Google Landmark Recognition training pipelines (Shallow CNN, Deep CNN, ViT Transformer, and CNN depth experiments).

Follow the instructions below to correctly install and manage these dependencies.

---

## Create and Activate a Virtual Environment (Recommended)
Creating a virtual environment isolates your dependencies and prevents conflicts.

### **Linux / macOS**

```python3 -m venv venv```
```source venv/bin/activate```

### Windows

```python -m venv venv```
```venv\Scripts\activate```

Install all dependencies from requirements.txt
Once the environment is activated, run: pip install -r requirements.txt or pip freeze > requirements.txt


This installs every library needed for:
- Shallow CNN
- Deep ResNet-based CNN
- ViT Transformer model
- Autoencoder pretraining
- Data loading and augmentation
- Logging & evaluation tools


## `demo.py` — Pipeline Demonstration Script

The `demo.py` script provides a minimal demonstration of how the **Google Landmarks Recognition** pipeline operates using this project's modules.  
It serves as a quick functional check to ensure that:

- the dataset loads correctly  
- transforms are applied properly  
- the model initializes without errors  
- a forward pass runs end-to-end  

This script **does not train** any models—it only performs inference on a small sample batch.

---

1. **Loads a Small Batch of Images**  
   Uses `make_loaders()` from `src.data` to create train/validation DataLoaders and fetch one batch from the training set.

2. **Initializes the ShallowCNN Model**  
   Constructs a lightweight CNN from `src.models` configured for the 50-class demo split.

3. **Runs a Forward Pass**  
   Executes a single forward pass under `torch.no_grad()` and prints the output tensor shapes.

4. **Displays Inputs and Outputs**  
   Prints:
   - input batch shape  
   - label indices  
   - logits shape  
   - example logits  
   - predicted class indices  

---

### **Example Output **

```
===== SAMPLE INPUT BATCH =====
Image batch shape: torch.Size([4, 3, 224, 224])
Labels: [12, 4, 7, 19]

===== MODEL OUTPUT =====
Logits shape: torch.Size([4, 50])
Sample logits row 0: tensor([...])

===== PREDICTIONS =====
Predicted classes: [12, 3, 7, 22]

Demo complete!
```

---

### **Modules Used**

| Module | Purpose |
|--------|---------|
| `src.data.make_loaders` | Loads GLDv2 dataset + DataLoader creation |
| `src.transforms.get_transforms` | Applies augmentation and normalization |
| `src.models.ShallowCNN` | Lightweight CNN architecture for testing |

---

### **Scripts Purpose**

- Verifies correct project setup and dataset paths  
- Confirms transforms and preprocessing work  
- Checks model initialization and forward connectivity  
- Provides a simple reference for writing custom scripts  

---













