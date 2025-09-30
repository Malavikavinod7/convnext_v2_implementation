import torch.nn as nn
import torchvision.models as models

def get_model(num_classes: int):
    # Load ConvNeXt-Tiny backbone
    model = models.convnext_tiny(weights="IMAGENET1K_V1")

    # Get input features of classifier
    in_features = model.classifier[2].in_features

    # Replace classifier with Dropout + Linear
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model

