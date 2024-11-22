# dataloader.py

import torch
from torchvision import datasets, transforms

def get_transforms():
    """Define and return the image transformations."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_train_dataset(root="/mnt/database/unaug_images/train"):
    """Initialize and return the training dataset."""
    transform = get_transforms()
    return datasets.ImageFolder(root=root, transform=transform)

def get_val_dataset(root="/mnt/database/unaug_images/val"):
    """Initialize and return the validation dataset."""
    transform = get_transforms()
    return datasets.ImageFolder(root=root, transform=transform)

def get_test_dataset(root="/mnt/database/unaug_images/test"):
    """Initialize and return the test dataset."""
    transform = get_transforms()
    return datasets.ImageFolder(root=root, transform=transform)

