from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class SimpleCNN(nn.Module):
    """
    Simple CNN for candlestick pattern classification.
    
    Uses ResNet18 backbone with proper initialization for financial pattern recognition.
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        """
        Args:
            num_classes: Number of pattern classes (default: 5)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(SimpleCNN, self).__init__()
        
        # Use ResNet18 as backbone with proper weights parameter
        if pretrained:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace the final layer for our number of classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize the new classifier layer properly
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0)
        
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

def create_cnn_model(
    model_type: str = "simple",
    num_classes: int = 5,
    pretrained: bool = True
) -> nn.Module:
    """
    Create a CNN model for candlestick classification.
    
    Args:
        model_type: "simple" (ResNet18) or "custom" (small CNN)
        num_classes: Number of pattern classes
        pretrained: Whether to use pretrained weights (for simple model)
        
    Returns:
        nn.Module: The created model
    """
    if model_type == "simple":
        model = SimpleCNN(num_classes=num_classes, pretrained=pretrained)
        print(f"Created SimpleCNN with {num_classes} classes")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model
