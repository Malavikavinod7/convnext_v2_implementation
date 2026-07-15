from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms


def get_datasets(img_size: int = 224, data_dir: str | None = None) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Load train and validation datasets from a folder-based structure."""
    base_dir = Path(data_dir or os.environ.get("DATA_DIR", Path(__file__).resolve().parent / "data_split"))
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transforms)

    return train_dataset, val_dataset

