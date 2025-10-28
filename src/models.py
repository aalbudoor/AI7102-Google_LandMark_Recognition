# src/models.py
import torch.nn as nn
import timm
import torchvision.models as models
import logging
import time

# ----------------------------------------------------------------------
# Configure logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shallow CNN
# ----------------------------------------------------------------------
class CNNShallow(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        start = time.time()
        logger.info("🧱 Initializing Shallow CNN model")

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 112 * 112, num_classes)  # for 224x224 input
        )

        elapsed = time.time() - start
        logger.info(f"✅ Shallow CNN initialized with {num_classes} classes in {elapsed:.2f}s")

    def forward(self, x):
        logger.debug("Forward pass through Shallow CNN")
        return self.net(x)


# ----------------------------------------------------------------------
# Deep CNN
# ----------------------------------------------------------------------
class CNNDeep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        start = time.time()
        logger.info("🧱 Initializing Deep CNN model")

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)

        elapsed = time.time() - start
        logger.info(f"✅ Deep CNN initialized with {num_classes} classes in {elapsed:.2f}s")

    def forward(self, x):
        logger.debug("Forward pass through Deep CNN")
        f = self.features(x).flatten(1)
        return self.head(f)


# ----------------------------------------------------------------------
# Vision Transformer (ViT)
# ----------------------------------------------------------------------
class ViTTiny(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        start = time.time()
        logger.info("🧱 Initializing Vision Transformer (ViT-Tiny) model")

        try:
            self.backbone = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=pretrained,
                num_classes=num_classes
            )
            elapsed = time.time() - start
            logger.info(
                f"✅ Vision Transformer initialized "
                f"({'pretrained' if pretrained else 'random'}) in {elapsed:.2f}s"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize ViT model: {e}")
            raise

    def forward(self, x):
        logger.debug("Forward pass through ViT-Tiny")
        return self.backbone(x)
