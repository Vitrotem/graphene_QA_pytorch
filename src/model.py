"""Model factory for TEM grid graphene classification."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
