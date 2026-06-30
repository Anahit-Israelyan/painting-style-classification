from pathlib import Path
from typing import Tuple

import torch
from PIL import ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    IMG_SIZE,
    BATCH_SIZE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
)


class PadToSquare:
    """
    Pad a PIL image to square shape while preserving aspect ratio.

    This avoids geometric distortion before resizing to IMG_SIZE x IMG_SIZE.
    """

    def __init__(self, fill=(0, 0, 0)):
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top

        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def get_train_transforms(img_size: int = IMG_SIZE):
    """
    Training transformations.

    We use light augmentation because the dataset is small.
    """
    return transforms.Compose(
        [
            PadToSquare(fill=(0, 0, 0)),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE):
    """
    Validation/test transformations.

    No random augmentation is used for evaluation.
    """
    return transforms.Compose(
        [
            PadToSquare(fill=(0, 0, 0)),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_datasets():
    """
    Load ImageFolder datasets from data/processed/.
    """
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_eval_transforms(),
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_eval_transforms(),
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.

    num_workers=0 is safest on Windows.
    In Colab, we can later use num_workers=2.
    """
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(train_dataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    This helps with the class imbalance observed in EDA.
    """
    targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    num_classes = len(train_dataset.classes)

    class_counts = torch.bincount(targets, minlength=num_classes).float()
    total = class_counts.sum()

    class_weights = total / (num_classes * class_counts)

    return class_weights