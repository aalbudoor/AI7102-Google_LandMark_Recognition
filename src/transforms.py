# src/transforms.py
from torchvision import transforms

def get_transforms(img_size=128):
    """
    Returns the image transformations for:
      - training (with augmentation)
      - validation/testing (no augmentation)

    img_size: final image size fed into the model (e.g., 128, 224)
    """

    # TRAINING TRANSFORMS (AUGMENTATION)
    # Purpose:
    #   - make the model more robust by simulating variations
    #   - helps prevent overfitting

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)), # randomly crop + resize → simulates zooming in/out
        transforms.RandomHorizontalFlip(p=0.5), # flip image left-right with 50% chance
        transforms.RandomRotation(15), # small random rotation


        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # random brightness/contrast/saturation changes (lighting)

        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), # small shifts and affine distortions

        transforms.ToTensor(), # convert PIL → PyTorch tensor

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize to ImageNet stats (for CNN/ViT pretrained models)

    ])


    # VALIDATION TRANSFORMS (NO AUGMENTATION)
    # Purpose:
    #   - keep validation data clean and consistent
    #   - only resize + normalize

    val_tfms = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return {"train": train_tfms, "val": val_tfms} # return dict used in training loop
