"""模型定义：ImageNet 预训练 ResNet18 + 7 类分类头。"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from labels import NUM_CLASSES


def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    return model
