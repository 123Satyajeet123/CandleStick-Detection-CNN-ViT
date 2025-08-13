from __future__ import annotations
import torch
import torch.nn as nn
import os
import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detect_candlestick.training.datasets import create_dataloaders
from detect_candlestick.models.cnn import create_cnn_model
from detect_candlestick.models.vit import create_vit_model

class ModelComparator:
    """Compare CNN and ViT model performance."""
    
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
        
        # Setup data
        self.setup_data()
        
        # Results storage
        self.results = {}
        
    def setup_data(self):
        """Setup data loaders."""
        print("\nSetting up data...")
        
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
        
    def load_model(self, model_path: str, model_type: str):
        """Load a trained model."""
        print(f"\nLoading {model_type} model from {model_path}...")
        
        # Load checkpoint with compatibility fix for newer PyTorch versions
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Standard loading failed, trying weights_only=True...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Create model
        if model_type == "cnn":
            model = create_cnn_model(
                model_type=self.config['model']['type'],
                num_classes=len(self.label_to_idx),
                pretrained=False
            )
        elif model_type == "vit":
            vit_config = self.config.get('vit', {})
            model = create_vit_model(
                model_type=vit_config.get('type', 'base'),
                num_classes=len(self.label_to_idx),
                **vit_config.get('params', {})
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"{model_type.upper()} model loaded successfully")
        print(f" Best validation F1: {checkpoint['best_val_f1']:.4f}")
        print(f" Trained for {checkpoint['epoch']} epochs")
        
        return model, checkpoint
    
    def evaluate_model(self, model: nn.Module, model_name: str, dataloader, split_name: str):
        """Evaluate a model on a specific dataset split."""
        print(f"\nEvaluating {model_name} on {split_name} set...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'f1_macro': f1_score(all_labels, all_predictions, average='macro'),
            'f1_weighted': f1_score(all_labels, all_predictions, average='weighted'),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision_macro': precision_score(all_labels, all_predictions, average='macro'),
            'recall_macro': recall_score(all_labels, all_predictions, average='macro'),
            'precision_weighted': precision_score(all_labels, all_predictions, average='weighted'),
            'recall_weighted': recall_score(all_labels, all_predictions, average='weighted')
        }
        
        # Per-class metrics
        per_class_f1 = f1_score(all_labels, all_predictions, average=None)
        for i, label in enumerate(self.label_to_idx.keys()):
            metrics[f'f1_{label}'] = per_class_f1[i]
        
        # Store results
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][split_name] = {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        # Print results
        print(f"{model_name.upper()} {split_name.capitalize()} Results:")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        
        return metrics
    
    def compare_models(self):
        """Compare CNN and ViT models."""
        print("\nStarting model comparison...")
        
        # Load models
        cnn_path = os.path.join(self.config['model_saving']['save_dir_cnn'], 'best_model.pth')
        vit_path = os.path.join(self.config['model_saving']['save_dir_vit'], 'best_model.pth')
        
        if not os.path.exists(cnn_path):
            print(f"CNN model not found at {cnn_path}")
            return
        
        if not os.path.exists(vit_path):
            print(f"ViT model not found at {vit_path}")
            return
        
        # Load CNN model
        cnn_model, cnn_checkpoint = self.load_model(cnn_path, "cnn")
        
        # Load ViT model
        vit_model, vit_checkpoint = self.load_model(vit_path, "vit")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("MODEL COMPARISON ON TEST SET")
        print("="*60)
        
        cnn_metrics = self.evaluate_model(cnn_model, "CNN", self.test_loader, "test")
        vit_metrics = self.evaluate_model(vit_model, "ViT", self.test_loader, "test")
        
        # Compare results
        self.print_comparison(cnn_metrics, vit_metrics)
        
        # Create comparison plots
        self.create_comparison_plots()
        
        # Save detailed results
        self.save_comparison_results()
        
        print("\nModel comparison complete!")
        
    def print_comparison(self, cnn_metrics: dict, vit_metrics: dict):
        """Print detailed comparison between models."""
        print("\n" + "="*80)
        print("DETAILED MODEL COMPARISON")
        print("="*80)
        
        comparison_data = []
        for metric in ['f1_macro', 'f1_weighted', 'accuracy', 'precision_macro', 'recall_macro']:
            cnn_val = cnn_metrics[metric]
            vit_val = vit_metrics[metric]
            diff = vit_val - cnn_val
            winner = "ViT" if vit_val > cnn_val else "CNN" if cnn_val > vit_val else "Tie"
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'CNN': f"{cnn_val:.4f}",
                'ViT': f"{vit_val:.4f}",
                'Difference': f"{diff:+.4f}",
                'Winner': winner
            })
        
        # Print comparison table
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Overall winner
        cnn_avg = np.mean([cnn_metrics['f1_macro'], cnn_metrics['accuracy']])
        vit_avg = np.mean([vit_metrics['f1_macro'], vit_metrics['accuracy']])
        
        print(f"\nOVERALL WINNER:")
        if vit_avg > cnn_avg:
            print(f"ViT wins with average score: {vit_avg:.4f} > CNN: {cnn_avg:.4f}")
        elif cnn_avg > vit_avg:
            print(f"CNN wins with average score: {cnn_avg:.4f} > ViT: {vit_avg:.4f}")
        else:
            print(f"It's a tie! Both models have average score: {cnn_avg:.4f}")
    
    def create_comparison_plots(self):
        """Create comparison plots."""
        print("\nCreating comparison plots...")
        
        # Create comparison directory
        comparison_dir = "models/comparison"
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 1. Metrics comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['F1 (Macro)', 'F1 (Weighted)', 'Accuracy', 'Precision', 'Recall']
        cnn_values = [
            self.results['CNN']['test']['metrics']['f1_macro'],
            self.results['CNN']['test']['metrics']['f1_weighted'],
            self.results['CNN']['test']['metrics']['accuracy'],
            self.results['CNN']['test']['metrics']['precision_macro'],
            self.results['CNN']['test']['metrics']['recall_macro']
        ]
        vit_values = [
            self.results['ViT']['test']['metrics']['f1_macro'],
            self.results['ViT']['test']['metrics']['f1_weighted'],
            self.results['ViT']['test']['metrics']['accuracy'],
            self.results['ViT']['test']['metrics']['precision_macro'],
            self.results['ViT']['test']['metrics']['recall_macro']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, cnn_values, width, label='CNN', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, vit_values, width, label='ViT', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('CNN vs ViT Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (cnn_val, vit_val) in enumerate(zip(cnn_values, vit_values)):
            ax.text(i - width/2, cnn_val + 0.01, f'{cnn_val:.3f}', ha='center', va='bottom')
            ax.text(i + width/2, vit_val + 0.01, f'{vit_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class F1 comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(self.label_to_idx.keys())
        cnn_class_f1 = [self.results['CNN']['test']['metrics'][f'f1_{cls}'] for cls in classes]
        vit_class_f1 = [self.results['ViT']['test']['metrics'][f'f1_{cls}'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax.bar(x - width/2, cnn_class_f1, width, label='CNN', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, vit_class_f1, width, label='ViT', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Candlestick Patterns')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'per_class_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to {comparison_dir}/")
    
    def save_comparison_results(self):
        """Save detailed comparison results to CSV."""
        comparison_dir = "models/comparison"
        
        # Create detailed results DataFrame
        results_data = []
        
        for model_name in ['CNN', 'ViT']:
            metrics = self.results[model_name]['test']['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(comparison_dir, 'detailed_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Detailed results saved to {csv_path}")

def main():
    """Main comparison function."""
    comparator = ModelComparator()
    comparator.compare_models()

if __name__ == "__main__":
    main()