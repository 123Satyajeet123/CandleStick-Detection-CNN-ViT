from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import sys
import yaml
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detect_candlestick.training.datasets import create_dataloaders
from detect_candlestick.models.vit import create_vit_model

class EarlyStopping:
    """Advanced early stopping with multiple metrics."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0005, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, score: float) -> bool:
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        return self.counter >= self.patience

class CandlestickViTTrainer:
    """Candlestick ViT trainer"""
    
    def __init__(self, config_path: str = "configs/training.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if self.config['hardware']['device'] == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config['hardware']['device'])
        
        print(f" Using device: {self.device}")
        
        # Initialize components
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.learning_rates = []
        
    def setup_data(self):
        """Setup data loaders"""
        print("\nSetting up data pipeline...")
        
        self.train_loader, self.val_loader, self.test_loader, self.label_to_idx = create_dataloaders(
            train_manifest="data/interim/train.csv",
            val_manifest="data/interim/val.csv",
            test_manifest="data/interim/test.csv",
            images_dir="data/images",
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            image_size=self.config['data']['image_size']
        )
        
        print(f"Number of classes: {len(self.label_to_idx)}")
        print(f"Label mapping: {self.label_to_idx}")
        
        # Calculate class weights for imbalanced dataset
        if self.config['training']['loss']['type'] == 'weighted_ce':
            train_dataset = self.train_loader.dataset
            self.class_weights = train_dataset.get_class_weights().to(self.device)
            print(f"Class weights: {self.class_weights}")

            self.weights = self._calculate_enhanced_weights()
            print(f"weights: {self.weights}")
        
    def _calculate_enhanced_weights(self):
        """Calculate class weights dynamically from the training dataset.

        Uses sqrt of inverse frequency and normalizes to number of classes.
        """
        # Get class counts aligned to label indices
        train_dataset = self.train_loader.dataset
        class_counts = train_dataset.get_class_counts().to(torch.float32).to(self.device)

        # Avoid divide-by-zero
        class_counts = class_counts.clamp(min=1)

        total_samples = class_counts.sum()
        inverse_freq = total_samples / class_counts

        # Reduce extremity of weights
        sqrt_weights = torch.sqrt(inverse_freq)

        # Normalize to sum to number of classes
        normalized_weights = sqrt_weights * len(class_counts) / sqrt_weights.sum()

        # Further emphasize historically weak classes (Hammer, SpinningTop)
        # Identify their indices via label_to_idx mapping
        hammer_idx = self.label_to_idx.get('Hammer')
        spinning_idx = self.label_to_idx.get('SpinningTop')
        if hammer_idx is not None:
            normalized_weights[hammer_idx] = normalized_weights[hammer_idx] * 1.5
        if spinning_idx is not None:
            normalized_weights[spinning_idx] = normalized_weights[spinning_idx] * 1.3

        return normalized_weights
        
    def setup_model(self):
        """Setup the optimized ViT model."""
        print("\nSetting up optimized ViT model...")
        
        # Get ViT-specific configuration
        vit_config = self.config.get('vit', {})
        model_type = vit_config.get('type', 'candlestick')
        
        self.model = create_vit_model(
            model_type=model_type,
            num_classes=len(self.label_to_idx),
            **vit_config.get('params', {})
        )
        self.model = self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Calculate model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"Model size: {model_size_mb:.1f} MB")
        
    def setup_training(self):
        """Setup training components."""
        print("\nSetting up training pipeline...")
        
        # GPU memory optimization for RTX 4050
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory: {torch.cuda.get_device_name(0)}")
            print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"Available VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        # loss function with focal loss option
        if self.config['training']['loss']['type'] == 'weighted_ce':
            # Use weights for better imbalance handling
            self.criterion = nn.CrossEntropyLoss(
                weight=self.weights,
                label_smoothing=self.config['training']['loss']['label_smoothing']
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config['training']['loss']['label_smoothing']
            )
        
        # Optimizer - ViT typically needs lower learning rate
        vit_lr = self.config['training'].get('vit_learning_rate', 3e-5)
        print(f"Using ViT learning rate: {vit_lr}")
        
        # optimizer with better parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(vit_lr),
            weight_decay=float(self.config['training']['vit_weight_decay']),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # learning rate scheduler
        if self.config['training']['scheduler']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=len(self.train_loader) * 10,  # Restart every 10 epochs
                T_mult=2,  # Double the restart interval
                eta_min=float(self.config['training']['scheduler']['min_lr'])
            )
        elif self.config['training']['scheduler']['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        
        # early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta'],
            mode='max'  # Monitor F1 score (higher is better)
        )

        # mixed precision training
        self.scaler = GradScaler() if self.config['hardware']['mixed_precision'] else None

        # memory efficiency optimizations
        if self.config['hardware'].get('memory_efficient', False) and torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("Enabled Flash Attention for memory efficiency")
            except:
                print("Flash Attention not available, using standard attention")
        
        # Create ViT model directory
        vit_save_dir = self.config['model_saving']['save_dir_vit']
        os.makedirs(vit_save_dir, exist_ok=True)
        
        print("Training pipeline ready!")
        
    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config['training']['gradient_clip']))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config['training']['gradient_clip']))
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Enhanced logging
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss, epoch_f1
    
    def validate_epoch(self) -> tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss, epoch_f1
    
    def save_model(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_f1': self.best_val_f1,
            'label_to_idx': self.label_to_idx,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_f1s': self.train_f1s,
            'val_f1s': self.val_f1s,
            'learning_rates': self.learning_rates,
            'class_weights': self.weights.cpu().numpy() if hasattr(self, 'weights') else None,
        }
        
        save_path = os.path.join(self.config['model_saving']['save_dir_vit'], filename)
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.config['model_saving']['save_dir_vit'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best ViT model saved! F1: {self.best_val_f1:.4f}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('ViT Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 plot
        ax2.plot(self.train_f1s, label='Train F1', color='green')
        ax2.plot(self.val_f1s, label='Val F1', color='orange')
        ax2.set_title('ViT Training and Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(self.learning_rates, label='Learning Rate', color='purple')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # Loss vs F1 correlation
        ax4.scatter(self.val_losses, self.val_f1s, alpha=0.6, color='red')
        ax4.set_xlabel('Validation Loss')
        ax4.set_ylabel('Validation F1 Score')
        ax4.set_title('Loss vs F1 Correlation')
        ax4.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.config['model_saving']['save_dir_vit'], 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path}")
    
    def train(self):
        """Training loop"""
        print(f"\nStarting ViT training for {self.config['training']['num_epochs']} epochs...")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Target: Beat CNN performance of 95.44% F1-score!")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch + 1
            
            # Training
            train_loss, train_f1 = self.train_epoch()
            
            # Validation
            val_loss, val_f1 = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {self.current_epoch}/{self.config['training']['num_epochs']}")
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # Performance comparison with CNN
            if val_f1 > 0.9544:
                print(f"BEATING CNN! Current F1: {val_f1:.4f} > CNN: 0.9544")
            else:
                gap = 0.9544 - val_f1
                print(f"Gap to CNN: {gap:.4f} (CNN: 0.9544)")
            
            # GPU memory monitoring for RTX 4050
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_model('best_model.pth', is_best=True)
                
                # Check if we've beaten CNN
                if val_f1 > 0.9544:
                    print(f"NEW RECORD! ViT beats CNN: {val_f1:.4f} > 0.9544")
            
            # Save checkpoint
            if (epoch + 1) % self.config['model_saving']['save_frequency'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping check
            if self.early_stopping(val_f1):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Final performance comparison
        if self.best_val_f1 > 0.9544:
            print(f"SUCCESS! ViT beats CNN: {self.best_val_f1:.4f} > 0.9544")
        else:
            print(f"ViT performance: {self.best_val_f1:.4f} (CNN: 0.9544)")
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate on test set
        self.evaluate_test()
    
    def evaluate_test(self):
        """Evaluate the model on the test set"""
        print("\nEvaluating ViT on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_f1 = f1_score(all_labels, all_predictions, average='macro')
        test_accuracy = accuracy_score(all_labels, all_predictions)
        test_precision = precision_score(all_labels, all_predictions, average='macro')
        test_recall = recall_score(all_labels, all_predictions, average='macro')
        
        print(f"ViT Test Results:")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        
        # Performance comparison
        if test_f1 > 0.9544:
            improvement = test_f1 - 0.9544
            print(f"BEATS CNN by {improvement:.4f} F1 points!")
        else:
            gap = 0.9544 - test_f1
            print(f"Gap to CNN: {gap:.4f} F1 points")
        
        # Per-class analysis
        per_class_f1 = f1_score(all_labels, all_predictions, average=None)
        print(f"\nPer-Class F1 Scores:")
        for i, (label, f1) in enumerate(zip(self.label_to_idx.keys(), per_class_f1)):
            print(f"  {label}: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.label_to_idx.keys()),
                   yticklabels=list(self.label_to_idx.keys()))
        plt.title('ViT Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        save_path = os.path.join(self.config['model_saving']['save_dir_vit'], 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")

def main():
    """Main training function."""
    trainer = CandlestickViTTrainer()
    trainer.train()

if __name__ == "__main__":
    main()

