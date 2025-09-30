import os
import shutil
import random
from pathlib import Path

def split_dataset(data_dir, output_dir, split_ratio=0.8, seed=42):
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    for cls in classes:
        class_dir = data_dir / cls
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Make train/val folders
        train_dir = output_dir / "train" / cls
        val_dir = output_dir / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(img, train_dir / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / img.name)

        print(f"{cls}: {len(train_images)} train, {len(val_images)} val")

if __name__ == "__main__":
    split_dataset(
        data_dir=r"C:\Users\DELL\OneDrive\Desktop\CONVNEXT_V2_IMPLEMENTATION\data",   # your current dataset
        output_dir=r"C:\Users\DELL\OneDrive\Desktop\CONVNEXT_V2_IMPLEMENTATION\data_split", # new structured dataset
        split_ratio=0.7   
    )
