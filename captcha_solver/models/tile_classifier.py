"""ResNet-18 based tile classifier for CAPTCHA image classification."""

import torch.nn as nn
import torchvision.models as models


class TileClassifier(nn.Module):
    """CNN tile classifier using ResNet-18 backbone.

    Classifies individual CAPTCHA tiles into one of num_classes categories.
    Uses pretrained ImageNet weights for transfer learning.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
