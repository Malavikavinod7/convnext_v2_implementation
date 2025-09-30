# train.py
import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import get_datasets
from torch import amp
from tqdm import tqdm
import random

# ---------------------------
# Mixup + CutMix utilities
# ---------------------------
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(torch.tensor(1. - lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = random.randint(0, W)
    cy = random.randint(0, H)

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

# ---------------------------
# Training function
# ---------------------------
def train_model(num_epochs=30, batch_size=16, lr=1e-4, device=None, log_dir="outputs"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load datasets
    train_dataset, val_dataset = get_datasets(img_size=224)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"DataLoaders ready. Train: {len(train_loader)}, Val: {len(val_loader)}")

    # Load ConvNeXt Base with a more complex head
    model = models.convnext_base(weights="IMAGENET1K_V1")
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(1024, len(train_dataset.classes))
    )
    model = model.to(device)

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    # AMP scaler for mixed-precision training
    scaler = amp.GradScaler()

    # Logging setup
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    patience, patience_counter = 7, 0  # Increased patience for early stopping

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            # Apply Mixup or CutMix randomly
            if random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels)
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels)

            optimizer.zero_grad(set_to_none=True)
            # ✅ FIXED HERE
            with amp.autocast(device_type=device):
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
            train_bar.set_postfix({"Loss": f"{train_loss/total:.4f}", "Acc": f"{correct/total:.4f}"})

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                # ✅ FIXED HERE
                with amp.autocast(device_type=device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                val_bar.set_postfix({"Loss": f"{val_loss/total:.4f}", "Acc": f"{correct/total:.4f}"})

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total

        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
              f"Time: {elapsed_time:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc])

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete! Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()
