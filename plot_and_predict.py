import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

# -----------------------------
# Paths
# -----------------------------
val_dir = r"C:\Users\DELL\OneDrive\Desktop\convnext_v2_implementation\data_split\val"
output_dir = r"C:\Users\DELL\OneDrive\Desktop\convnext_v2_implementation\outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Dataset loader
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = val_dataset.classes

# -----------------------------
# Load trained model (ConvNeXt Base)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join(output_dir, "best_model.pth")

# Rebuild ConvNeXt with same head as training
model = models.convnext_base(weights="IMAGENET1K_V1")
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 1024),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(1024, len(class_names))
)

# Load weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Inference
# -----------------------------
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

# -----------------------------
# Save predictions.csv
# -----------------------------
df = pd.DataFrame({
    "true": [class_names[i] for i in all_labels],
    "pred": [class_names[i] for i in all_preds],
    "confidence": np.max(all_probs, axis=1)
})
df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# -----------------------------
# Classification report
# -----------------------------
report = classification_report(all_labels, all_preds, target_names=class_names)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# -----------------------------
# Accuracy/Loss curves
# -----------------------------
log_path = os.path.join(output_dir, "train_log.csv")
if os.path.exists(log_path):
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
    plt.savefig(os.path.join(output_dir, "accuracy_loss.png"))
    plt.close()

# -----------------------------
# Confidence histogram
# -----------------------------
plt.hist(df["confidence"], bins=20, color="purple")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "confidence_histogram.png"))
plt.close()

# -----------------------------
# ROC Curves
# -----------------------------
n_classes = len(class_names)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc(fpr,tpr):.2f})")
plt.legend()
plt.title("ROC Curves")
plt.savefig(os.path.join(output_dir, "roc_curves.png"))
plt.close()

# -----------------------------
# PR Curves
# -----------------------------
for i in range(n_classes):
    prec, rec, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
    plt.plot(rec, prec, label=class_names[i])
plt.legend()
plt.title("PR Curves")
plt.savefig(os.path.join(output_dir, "pr_curves.png"))
plt.close()

# -----------------------------
# t-SNE visualization
# -----------------------------
features = TSNE(n_components=2, random_state=42).fit_transform(all_probs)
plt.scatter(features[:, 0], features[:, 1], c=all_labels, cmap="tab20", s=10)
plt.colorbar()
plt.title("t-SNE of Validation Set")
plt.savefig(os.path.join(output_dir, "tsne.png"))
plt.close()

# -----------------------------
# Calibration curve
# -----------------------------
prob_true, prob_pred = calibration_curve(all_labels == all_preds,
                                         df["confidence"], n_bins=10)
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.title("Calibration Curve")
plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
plt.close()

# -----------------------------
# Correct & Wrong Predictions
# -----------------------------
def save_image_grid(indices, title, filename):
    imgs = []
    for idx in indices[:25]:  # Take max 25 samples
        img, _ = val_dataset[idx]
        imgs.append(img)
    grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

correct_idx = np.where(all_labels == all_preds)[0]
wrong_idx = np.where(all_labels != all_preds)[0]

if len(correct_idx) > 0:
    save_image_grid(correct_idx, "Correct Predictions", "correct_predictions.png")

if len(wrong_idx) > 0:
    save_image_grid(wrong_idx, "Wrong Predictions", "wrong_predictions.png")

print("✅ All plots and reports saved in:", output_dir)


