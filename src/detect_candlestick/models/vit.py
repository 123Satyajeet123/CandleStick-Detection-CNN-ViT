from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

class MLPBlock(nn.Sequential):
    """MLP Block for Transformer Encoder"""
    
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.3):
        super().__init__(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

class EncoderBlock(nn.Module):
    """Single Encoder Block"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, batch_first: bool = True, dropout: float = 0.3):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)
        
        # Add residual connection scaling for better training
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with scaled residual connection
        z1 = self.ln_1(x)
        attn_out = self.self_attention(z1, z1, z1, need_weights=False)[0]
        z1 = self.scale * attn_out + x
        z1 = self.dropout(z1)
        
        # MLP with scaled residual connection
        z2 = self.ln_2(z1)
        mlp_out = self.mlp(z2)
        z2 = self.scale * mlp_out + z1
        
        return z2

class Encoder(nn.Module):
    """Stacked Encoder"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Add layer normalization at the end
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

def pos_encoding(seq_length: int, dim_size: int) -> torch.Tensor:
    """Positional encoding"""
    p = torch.zeros((seq_length, dim_size))
    for k in range(seq_length):
        for i in range(int(dim_size / 2)):
            p[k, 2 * i] = torch.sin(torch.tensor(k / (10000 ** (2 * i / dim_size))))
            p[k, 2 * i + 1] = torch.cos(torch.tensor(k / (10000 ** (2 * i / dim_size))))
    return p

class CandlestickViT(nn.Module):
    """ViT for candlestick pattern classification"""
    
    def __init__(self, 
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 5,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_dim: int = 2048,
                 num_layers: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        
        # Calculate dimensions
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3
        
        # patch embedding
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # initialize patch embedding weights
        nn.init.kaiming_normal_(self.patch.weight, mode='fan_out', nonlinearity='relu')
        if self.patch.bias is not None:
            nn.init.constant_(self.patch.bias, 0)
        
        # Class token for classification
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.class_token, std=0.02)
        
        # Positional embedding
        self.register_buffer("positional_embedding", pos_encoding(self.num_patches + 1, embed_dim))
        
        # dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # layer norms
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Encoder stack - deeper for complex patterns
        self.encoder = Encoder(embed_dim, num_heads, mlp_dim, num_layers, dropout)
        
        #classification head for imbalanced classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5),  # less dropout in final layers
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CandlestickViT created with {total_params:,} parameters")
        print(f" - Embedding dim: {embed_dim}")
        print(f" - Number of heads: {num_heads}")
        print(f" - MLP dim: {mlp_dim}")
        print(f" - Number of layers: {num_layers}")
        print(f" - Patch size: {patch_size}x{patch_size}")
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create patches using conv2d approach
        x = self.patch(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # add class token
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # add positional embedding
        x = x + self.positional_embedding
        
        # apply dropout
        x = self.emb_dropout(x)
        
        # pass through encoder
        x = self.encoder(x)
        
        # extract class token for classification
        x = x[:, 0, :]  # (B, embed_dim)
        
        # Enhanced classification head
        x = self.classifier(x)
        
        return x

class LightweightViT(nn.Module):
    """Lightweight ViT optimized for RTX 4050 (6GB VRAM) - balanced performance."""
    
    def __init__(self, 
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 5,
                 embed_dim: int = 384,
                 num_heads: int = 6,
                 mlp_dim: int = 1536,
                 num_layers: int = 6,
                 dropout: float = 0.2):
        super().__init__()
        
        # Calculate dimensions
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3
        
        # Patch creation
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.patch.weight, mode='fan_out', nonlinearity='relu')
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.class_token, std=0.02)
        
        # Positional embedding
        self.register_buffer("positional_embedding", pos_encoding(self.num_patches + 1, embed_dim))
        
        # Dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Encoder stack
        self.encoder = Encoder(embed_dim, num_heads, mlp_dim, num_layers, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightweightViT created with {total_params:,} parameters")
        print(f"   - Embedding dim: {embed_dim}")
        print(f"   - Number of heads: {num_heads}")
        print(f"   - MLP dim: {mlp_dim}")
        print(f"   - Number of layers: {num_layers}")
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create patches
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.positional_embedding
        
        # Apply dropout
        x = self.emb_dropout(x)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Extract class token for classification
        x = x[:, 0, :]
        
        # Classification head
        x = self.classifier(x)
        
        return x

def create_vit_model(
    model_type: str = "candlestick",
    num_classes: int = 5,
    image_size: int = 224,
    **kwargs
) -> nn.Module:
    """
    Create a ViT model for candlestick classification - optimized to beat CNN.
    
    Args:
        model_type: "candlestick" (best performance), "lightweight" (RTX 4050 optimized)
        num_classes: Number of pattern classes
        image_size: Input image size
        **kwargs: Additional arguments for model configuration
        
    Returns:
        nn.Module: The created model
    """
    if model_type == "candlestick":
        # Best performance model - designed to beat CNN
        model = CandlestickViT(
            image_size=image_size,
            num_classes=num_classes,
            embed_dim=768,
            num_heads=12,
            mlp_dim=2048,
            num_layers=12,
            dropout=0.3,
            **kwargs
        )
        print(f"Created CandlestickViT")
        
    elif model_type == "lightweight":
        # RTX 4050 optimized
        model = LightweightViT(
            image_size=image_size,
            num_classes=num_classes,
            embed_dim=384,
            num_heads=6,
            mlp_dim=1536,
            num_layers=6,
            dropout=0.2,
            **kwargs
        )
        print(f"Created LightweightViT")
        
    elif model_type == "base":
        # Original repository architecture
        model = CandlestickViT(
            image_size=image_size,
            num_classes=num_classes,
            embed_dim=1024,
            num_heads=8,
            mlp_dim=2056,
            num_layers=3,
            dropout=0.25,
            **kwargs
        )
        print(f"Created Base ViT")

    elif model_type == "deit_tiny_pretrained":
        # Pretrained DeiT-Tiny from timm (fast, stable fine-tuning)
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm is required for deit_tiny_pretrained. Install with `uv add timm`." ) from e
        model = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=True,
            num_classes=num_classes
        )
        print("Loaded pretrained DeiT-Tiny (patch16, 224) from timm")

    elif model_type == "vit_small_pretrained":
        # Pretrained ViT-Small from timm (more capacity)
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm is required for vit_small_pretrained. Install with `uv add timm`." ) from e
        model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=num_classes
        )
        print("Loaded pretrained ViT-Small (patch16, 224) from timm")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model
