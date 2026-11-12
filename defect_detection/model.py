import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, pretrained: bool = True):
    """EfficientNet-B0 transfer-learning classifier for apple image classes."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
