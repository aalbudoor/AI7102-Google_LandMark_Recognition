import torch
import torch.nn as nn
import torchvision.models as models
import timm
import logging
import time

# Basic logging setup (so we can see model initialization info)
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# SHALLOW AUTOENCODER (used BEFORE classification)
# Purpose: learn simple features from images WITHOUT labels.
# Later, we copy the encoder part into the classifier.
# ---------------------------------------------------------
class ShallowAutoencoder(nn.Module):
    """Used only for pretraining on unlabeled images"""
    def __init__(self):
        super().__init__()
        # Encoder: a tiny CNN that compresses the image
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),nn.MaxPool2d(2))

        # Decoder: tries to rebuild the original image
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 3, 2, stride=2), nn.Sigmoid())


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




# The CNN CNN CLASSIFIER. This is the Autoencoder encoder + classifier head. If pretrained_path is given → load weights learned earlier.
class ShallowCNN(nn.Module):
    """Classifier version of the shallow CNN"""
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        start = time.time()
        logger.info("Initializing Shallow CNN model")

        # This is the SAME architecture as the Autoencoder encoder.
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),nn.MaxPool2d(2))

        # Classifier head → turns features into class probabilities
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(32 * 112 * 112, num_classes)) # 112 because 224/2 after MaxPool



        # If we have pretrained weights, load them
        if pretrained_path:
            try:
                weights = torch.load(pretrained_path, map_location="cpu")
                self.features.load_state_dict(weights)
                logger.info(f"Loaded pretrained encoder from {pretrained_path}")
            except Exception as e:
                logger.warning(f"Could not load pretrained encoder: {e}")

        elapsed = time.time() - start
        logger.info(f"Shallow CNN ready in {elapsed:.2f}s")

    def forward(self, x):
        return self.classifier(self.features(x))




# ---------------------------------------------------------
# DEEP CNN (RESNET18 BACKBONE)
# Idea:
#   - Use pretrained ResNet as a powerful feature extractor.
#   - Add extra convolution layers to specialize on our dataset.
#   - Add classification head.
# This is a more serious model than ShallowCNN.
# ---------------------------------------------------------
class DeepCNNPretrained(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        start = time.time()
        logger.info("Initializing Deep CNN (ResNet18 + extra conv layers)")

        # Load pretrained ResNet18 (minus the final layers)
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)


        # Keep everything except the final pooling + FC layer
        self.base_features = nn.Sequential(*list(backbone.children())[:-2])  # keep conv layers only

        in_feats = 512  # ResNet18 ends with 512 feature channels

        # Add extra convolutional layers on top
        self.extra_conv = nn.Sequential(nn.Conv2d(in_feats, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True),nn.Conv2d(512, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d((1, 1)))


        # Classifier head
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(256, 256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256, num_classes))

        elapsed = time.time() - start
        logger.info(f"Deep CNN initialized with extra conv layers in {elapsed:.2f}s")

    def forward(self, x):
        x = self.base_features(x)
        x = self.extra_conv(x)
        x = self.classifier(x)
        return x






# ---------------------------------------------------------
# VISION TRANSFORMER (ViT-B16)
# This uses the TIMM library to load a pretrained ViT model.
# ViT splits the image into patches → treats them like tokens.
# This is the strongest model in your pipeline.
# ---------------------------------------------------------
class ViTModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        start = time.time()
        logger.info("🧱 Initializing Vision Transformer (ViT-B16)")

        # Load ViT with the correct number of classes
        self.backbone = timm.create_model("vit_base_patch16_224",pretrained=pretrained,num_classes=num_classes)

        elapsed = time.time() - start
        logger.info(f"ViT initialized (pretrained={pretrained}) in {elapsed:.2f}s")

    def forward(self, x):
        return self.backbone(x)
