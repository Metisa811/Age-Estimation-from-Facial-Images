"""
Model architectures for the UTKFace age estimation project.

Both models share a ResNet18 (ImageNet-pretrained) backbone with a small
custom head, differing only in the head's output (a single scalar for
regression, or class logits for classification) and whether the backbone
is frozen.
"""

import torch.nn as nn
from torchvision import models


class AgeRegressionNet(nn.Module):
    """ResNet18 backbone + regression head predicting a single continuous age."""

    def __init__(self, freeze_backbone=False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)


class AgeClassifierNet(nn.Module):
    """ResNet18 backbone + classification head predicting an age-group bin."""

    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
