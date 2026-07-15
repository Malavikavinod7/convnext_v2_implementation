from __future__ import annotations

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int, pretrained: bool = True, model_name: str = "convnext_tiny"):
    """Create a ConvNeXt-based classifier with a custom head."""
    weights = "IMAGENET1K_V1" if pretrained else None

    if model_name.lower() == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
    elif model_name.lower() == "convnext_base":
        model = models.convnext_base(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )

    return model

