# src/models.py
import torch.nn as nn
import timm
import torchvision.models as models
import get_transforms as transforms

class CNNShallow(nn.Module):
    def __init__(self, num_classes):
        print("Initializing Shallow CNN")
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*112*112, num_classes)  # for 224x224 input
        )
    def forward(self, x): return self.net(x)

class CNNDeep(nn.Module):
    def __init__(self, num_classes):
        print("Initializing Deep CNN")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)
    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.head(f)

class ViTTiny(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        print("Initializing Vision Transformer (ViT)")
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
    def forward(self, x): return self.backbone(x)
