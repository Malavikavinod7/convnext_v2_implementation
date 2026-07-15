from __future__ import annotations

import argparse
import os
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ConvNeXt classifier")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data_split" / "val"), help="Validation dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"), help="Directory for reports and plots")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default="convnext_tiny", choices=["convnext_tiny", "convnext_base"])
    return parser.parse_args()


def run_evaluation(args=None):
    args = args or parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    class_names = val_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint or output_dir / "best_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=len(class_names), pretrained=False, model_name=args.model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    df = pd.DataFrame({
        "true": [class_names[i] for i in all_labels],
        "pred": [class_names[i] for i in all_preds],
        "confidence": np.max(all_probs, axis=1),
    })
    df.to_csv(output_dir / "predictions.csv", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    log_path = output_dir / "train_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        plt.figure()
        plt.plot(log_df["epoch"], log_df["train_acc"], label="Train Acc")
        plt.plot(log_df["epoch"], log_df["val_acc"], label="Val Acc")
        plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
        plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training/Validation Curves")
        plt.savefig(output_dir / "accuracy_loss.png")
        plt.close()

    plt.figure()
    plt.hist(df["confidence"], bins=20, color="purple")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.savefig(output_dir / "confidence_histogram.png")
    plt.close()

    n_classes = len(class_names)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc(fpr, tpr):.2f})")
    plt.legend()
    plt.title("ROC Curves")
    plt.savefig(output_dir / "roc_curves.png")
    plt.close()

    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        plt.plot(rec, prec, label=class_names[i])
    plt.legend()
    plt.title("PR Curves")
    plt.savefig(output_dir / "pr_curves.png")
    plt.close()

    if len(all_probs) > 2:
        features = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_probs) - 1)).fit_transform(all_probs)
        plt.scatter(features[:, 0], features[:, 1], c=all_labels, cmap="tab20", s=10)
        plt.colorbar()
        plt.title("t-SNE of Validation Set")
        plt.savefig(output_dir / "tsne.png")
        plt.close()
    else:
        plt.figure()
        plt.text(0.5, 0.5, "Not enough samples for t-SNE", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_dir / "tsne.png")
        plt.close()

    prob_true, prob_pred = calibration_curve(all_labels == all_preds, df["confidence"], n_bins=10)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve")
    plt.savefig(output_dir / "calibration_curve.png")
    plt.close()

    def save_image_grid(indices, title, filename):
        imgs = []
        for idx in indices[:25]:
            img, _ = val_dataset[idx]
            imgs.append(img)
        grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(title)
        plt.savefig(output_dir / filename)
        plt.close()

    correct_idx = np.where(all_labels == all_preds)[0]
    wrong_idx = np.where(all_labels != all_preds)[0]

    if len(correct_idx) > 0:
        save_image_grid(correct_idx, "Correct Predictions", "correct_predictions.png")
    if len(wrong_idx) > 0:
        save_image_grid(wrong_idx, "Wrong Predictions", "wrong_predictions.png")

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    run_evaluation()


