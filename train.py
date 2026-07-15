from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_datasets
from model import get_model


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(torch.tensor(1.0 - lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha: float = 1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha: float = 1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[index], lam


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ConvNeXt image classifier")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data_split"), help="Path to train/val folders")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"), help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--model-name", type=str, default="convnext_tiny", choices=["convnext_tiny", "convnext_base"])
    parser.add_argument("--pretrained", type=str, default="true", choices=["true", "false"], help="Use ImageNet-pretrained weights")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    return parser.parse_args()


def train_model(args=None):
    args = args or parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Training on: {device}")

    train_dataset, val_dataset = get_datasets(img_size=args.img_size, data_dir=args.data_dir)
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    print(f"DataLoaders ready. Train: {len(train_loader)}, Val: {len(val_loader)}")

    pretrained = args.pretrained.lower() == "true"
    model = get_model(num_classes=num_classes, pretrained=pretrained, model_name=args.model_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    scaler = amp.GradScaler(enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    patience, patience_counter = 7, 0

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            if random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels)
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels)

            optimizer.zero_grad(set_to_none=True)
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with amp.autocast(device_type=autocast_device, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (lam * preds.eq(y_a).sum().item() + (1 - lam) * preds.eq(y_b).sum().item())
            total += labels.size(0)
            train_bar.set_postfix({"Loss": f"{train_loss / total:.4f}", "Acc": f"{correct / total:.4f}"})

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                with amp.autocast(device_type=autocast_device, enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                val_bar.set_postfix({"Loss": f"{val_loss / total:.4f}", "Acc": f"{correct / total:.4f}"})

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total

        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{args.epochs}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | Time: {elapsed_time:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc])

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete! Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model()

