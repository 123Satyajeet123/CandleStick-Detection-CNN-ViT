from __future__ import annotations
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple

class CandlestickDataset(Dataset):
    """
    PyTorch Dataset for candlestick pattern classification.
    
    Loads images from the manifest and applies basic transforms.
    """
    
    def __init__(
        self, 
        manifest_path: str, 
        images_dir: str,
        transform=None,
        label_to_idx: Dict[str, int] = None
    ):
        """
        Args:
            manifest_path: Path to CSV manifest with image_path, label, ticker, date
            images_dir: Directory containing the image folders
            transform: Optional transforms to apply
            label_to_idx: Mapping from label names to indices
        """
        self.manifest_path = manifest_path
        self.images_dir = images_dir
        self.transform = transform
        
        # Load manifest
        self.df = pd.read_csv(manifest_path)
        print(f"Loaded {len(self.df)} samples from {manifest_path}")
        
        # Create label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(self.df['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Label mapping: {self.label_to_idx}")
        print(f"Class distribution:")
        print(self.df['label'].value_counts())
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            tuple: (image_tensor, label_idx)
        """
        row = self.df.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = row['label']
        label_idx = self.label_to_idx[label]
        
        return image, label_idx
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted loss."""
        class_counts = self.df['label'].value_counts()
        total_samples = len(self.df)
        
        weights = torch.zeros(len(self.label_to_idx))
        for label, count in class_counts.items():
            idx = self.label_to_idx[label]
            weights[idx] = total_samples / (len(class_counts) * count)
        
        return weights

def get_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get basic transforms for candlestick images.
    
    Args:
        image_size: Target image size (width=height)
    
    Returns:
        transforms.Compose: Basic transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_manifest: Path to train manifest CSV
        val_manifest: Path to validation manifest CSV
        test_manifest: Path to test manifest CSV
        images_dir: Directory containing images
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        image_size: Target image size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_to_idx)
    """
    # Get transforms
    transform = get_transforms(image_size)
    
    # Create datasets
    train_dataset = CandlestickDataset(train_manifest, images_dir, transform)
    
    # Use same label mapping for all datasets
    label_to_idx = train_dataset.label_to_idx
    
    val_dataset = CandlestickDataset(val_manifest, images_dir, transform, label_to_idx)
    test_dataset = CandlestickDataset(test_manifest, images_dir, transform, label_to_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader, label_to_idx 
