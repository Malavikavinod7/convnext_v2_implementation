from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(data_dir: str | Path, output_dir: str | Path, split_ratio: float = 0.8, seed: int = 42) -> None:
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    for cls in classes:
        class_dir = data_dir / cls
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        random.shuffle(images)

        split_idx = max(1, int(len(images) * split_ratio)) if len(images) > 1 else 0
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_dir = output_dir / "train" / cls
        val_dir = output_dir / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_dir / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / img.name)

        print(f"{cls}: {len(train_images)} train, {len(val_images)} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a class-based dataset into train/validation folders")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data"), help="Path to the source dataset")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "data_split"), help="Path for the split dataset")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Fraction used for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_dataset(args.data_dir, args.output_dir, split_ratio=args.split_ratio, seed=args.seed)

