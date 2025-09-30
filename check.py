import torch
import torchvision.models as models
import torch.nn as nn
import os

# Path to best model
model_path = os.path.join("outputs", "best_model.pth")

# Load ConvNeXt Base backbone
model = models.convnext_base(weights=None)  # No pretrained weights since we load ours

# Replace classifier head exactly as in training
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 1024),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(1024,  len(os.listdir("C:/Users/DELL/OneDrive/Desktop/convnext_v2_implementation/data_split/train")))
)

# Load the trained weights
model.load_state_dict(torch.load(model_path, map_location="cpu"))

print("✅ Model loaded successfully from:", model_path)
print(model)  # This will display the architecture for your screenshot


