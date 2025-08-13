from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class SimpleCNN(nn.Module):
    """
    MobileNetV3-Large classifier (efficient for real-time).
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        """
        Args:
            num_classes: Number of pattern classes (default: 5)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(SimpleCNN, self).__init__()
        
        # Use MobileNetV3-Large as backbone with proper weights parameter
        if pretrained:
            self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.mobilenet_v3_large(weights=None)
        
        # Replace the final classification layer for our number of classes
        # MobileNetV3 uses a `classifier` Sequential head; last layer is Linear
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize the new classifier layer properly
        nn.init.xavier_uniform_(self.backbone.classifier[-1].weight)
        nn.init.constant_(self.backbone.classifier[-1].bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.backbone(x)
        x = self.dropout(x)
        return x

class ResNet34CNN(nn.Module):
    """
    ResNet34 classifier (strong baseline on larger datasets).
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        # Load ResNet34 backbone
        if pretrained:
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.resnet34(weights=None)

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Regularization
        self.dropout = nn.Dropout(0.3)

        # Init classifier weights
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.dropout(x)
        return x

def create_cnn_model(
    model_type: str = "resnet34",
    num_classes: int = 5,
    pretrained: bool = True
) -> nn.Module:
    """
    Create a CNN model for candlestick classification.

    Args:
        model_type: one of {"resnet34", "resnet18", "mobilenet_v3"}
        num_classes: Number of pattern classes
        pretrained: Whether to use pretrained weights (for simple model)
        
    Returns:
        nn.Module: The created model
    """
    if model_type == "resnet34":
        model = ResNet34CNN(num_classes=num_classes, pretrained=pretrained)
        print(f"Created ResNet34 with {num_classes} classes")
    elif model_type == "resnet18":
        if pretrained:
            backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(backbone.fc.weight)
        nn.init.constant_(backbone.fc.bias, 0)
        model = nn.Sequential(backbone, nn.Dropout(0.3))
        print(f"Created ResNet18 with {num_classes} classes")
    elif model_type == "mobilenet_v3":
        model = SimpleCNN(num_classes=num_classes, pretrained=pretrained)
        print(f"Created MobileNetV3-Large with {num_classes} classes")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model
