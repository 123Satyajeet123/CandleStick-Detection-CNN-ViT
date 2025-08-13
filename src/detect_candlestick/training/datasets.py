from __future__ import annotations
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

    def get_class_counts(self) -> torch.Tensor:
        """Return class counts aligned to label indices.

        Returns:
            torch.Tensor: Tensor of length num_classes where position i is the
            count for class with index i (according to `label_to_idx`).
        """
        counts_series = self.df['label'].value_counts()
        counts = torch.zeros(len(self.label_to_idx), dtype=torch.long)
        for label, idx in self.label_to_idx.items():
            counts[idx] = int(counts_series.get(label, 0))
        return counts

def get_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """
    Build image transforms.

    - Training: light augmentations safe for candlesticks (no flips)
    - Eval: deterministic resize + normalize
    """
    if train:

        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
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
    train_transform = get_transforms(image_size, train=True)
    eval_transform = get_transforms(image_size, train=False)
    
    # Create datasets
    train_dataset = CandlestickDataset(train_manifest, images_dir, train_transform)
    
    # Use same label mapping for all datasets
    label_to_idx = train_dataset.label_to_idx
    
    val_dataset = CandlestickDataset(val_manifest, images_dir, eval_transform, label_to_idx)
    test_dataset = CandlestickDataset(test_manifest, images_dir, eval_transform, label_to_idx)
    
    # Create dataloaders
    # Build a WeightedRandomSampler to balance classes in training batches
    # Compute per-class counts
    class_counts_series = train_dataset.df['label'].value_counts()
    # Inverse frequency for class weights
    class_weight_map: Dict[str, float] = {
        label: (1.0 / float(class_counts_series[label])) for label in class_counts_series.index
    }
    # Per-sample weights from label
    sample_weights = [class_weight_map[row_label] for row_label in train_dataset.df['label']]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
