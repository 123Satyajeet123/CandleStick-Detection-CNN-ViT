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
from detect_candlestick.models.cnn import create_cnn_model

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class CandlestickTrainer:
    """Comprehensive trainer for candlestick pattern classification."""
    
    def __init__(self, config_path: str = "configs/training.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if self.config['hardware']['device'] == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config['hardware']['device'])
        
        print(f"Using device: {self.device}")
        
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
        
    def setup_data(self):
        """Setup data loaders."""
        print("\n📊 Setting up data...")
        
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
        
        # Calculate class weights for weighted loss
        if self.config['training']['loss']['type'] == 'weighted_ce':
            train_dataset = self.train_loader.dataset
            self.class_weights = train_dataset.get_class_weights().to(self.device)
            print(f"Class weights: {self.class_weights}")
        
    def setup_model(self):
        """Setup the CNN model."""
        print("\n🏗️ Setting up model...")
        
        self.model = create_cnn_model(
            model_type=self.config['model']['type'],
            num_classes=len(self.label_to_idx),
            pretrained=self.config['model']['pretrained']
        )
        self.model = self.model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def setup_training(self):
        """Setup training components."""
        print("\n⚙️ Setting up training...")
        
        # GPU memory optimization for RTX 4050
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory: {torch.cuda.get_device_name(0)}")
            print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"Available VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        # Loss function
        if self.config['training']['loss']['type'] == 'weighted_ce':
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.config['training']['loss']['label_smoothing']
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config['training']['loss']['label_smoothing']
            )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # Learning rate scheduler
        if self.config['training']['scheduler']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=float(self.config['training']['scheduler']['min_lr'])
            )
        elif self.config['training']['scheduler']['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta']
        )

        self.scaler = GradScaler() if self.config['hardware']['mixed_precision'] else None

        if self.config['hardware'].get('memory_efficient', False) and torch.cuda.is_available():
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("✅ Enabled Flash Attention for memory efficiency")
            except:
                print("⚠️ Flash Attention not available, using standard attention")
        
        # Create model directory
        os.makedirs(self.config['model_saving']['save_dir'], exist_ok=True)
        
    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
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
            
            # Logging
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss, epoch_f1
    
    def validate_epoch(self) -> tuple[float, float]:
        """Validate for one epoch."""
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
        """Save model checkpoint."""
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
        }
        
        save_path = os.path.join(self.config['model_saving']['save_dir'], filename)
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.config['model_saving']['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved! F1: {self.best_val_f1:.4f}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 plot
        ax2.plot(self.train_f1s, label='Train F1')
        ax2.plot(self.val_f1s, label='Val F1')
        ax2.set_title('Training and Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['model_saving']['save_dir'], 'training_history.png'))
        plt.close()
    
    def train(self):
        """Main training loop."""
        print(f"\n🎯 Starting training for {self.config['training']['num_epochs']} epochs...")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        
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
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {self.current_epoch}/{self.config['training']['num_epochs']}")
            print(f"  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # GPU memory monitoring for RTX 4050
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_model('best_model.pth', is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.config['model_saving']['save_frequency'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n✅ Training complete in {total_time/60:.1f} minutes!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate on test set
        self.evaluate_test()
    
    def evaluate_test(self):
        """Evaluate the model on the test set."""
        print("\n🧪 Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_f1 = f1_score(all_labels, all_predictions, average='macro')
        test_accuracy = accuracy_score(all_labels, all_predictions)
        test_precision = precision_score(all_labels, all_predictions, average='macro')
        test_recall = recall_score(all_labels, all_predictions, average='macro')
        
        print(f"Test Results:")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.label_to_idx.keys()),
                   yticklabels=list(self.label_to_idx.keys()))
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['model_saving']['save_dir'], 'confusion_matrix.png'))
        plt.close()

def main():
    """Main training function."""
    trainer = CandlestickTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
