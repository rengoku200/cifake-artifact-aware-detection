import torch
import torch.nn as nn
from torchvision import models

class VGG16CIFAKE(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        if pretrained:
            self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.vgg16(weights=None)

        # FEATURES = convolutional layers (backbone)
        # CLASSIFIER = original FC layers 
        in_features = self.backbone.classifier[6].in_features

        self.backbone.classifier[6] = nn.Linear(in_features, 2)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)  
