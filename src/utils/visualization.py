"""
Visualization utilities for ShelfRanger training results.
Generates plots from completed training without re-running training.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_training_results(results_csv_path="models/train/results.csv", 
                         output_dir="models/train/plots"):
    """
    Plot training metrics from results.csv file.
    
    Args:
        results_csv_path: Path to the results.csv file from training
        output_dir: Directory to save plots
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read training results
        if not Path(results_csv_path).exists():
            logger.error(f"Results file not found: {results_csv_path}")
            return
            
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()  # Remove any whitespace
        
        # Set up plotting style
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Training Losses Plot
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        loss_columns = [col for col in df.columns if 'loss' in col.lower()]
        for col in loss_columns:
            if col in df.columns:
                plt.plot(df.index + 1, df[col], label=col.replace('train/', ''), linewidth=2)
        plt.title('Training Losses', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Validation Metrics Plot
        plt.subplot(1, 3, 2)
        metric_columns = [col for col in df.columns if any(x in col for x in ['precision', 'recall', 'mAP50'])]
        for col in metric_columns:
            if col in df.columns and 'mAP50-95' not in col:  # Exclude mAP50-95 for clarity
                plt.plot(df.index + 1, df[col], label=col.replace('metrics/', '').replace('val/', ''), linewidth=2)
        plt.title('Validation Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 3. mAP Progression
        plt.subplot(1, 3, 3)
        map_columns = [col for col in df.columns if 'mAP' in col]
        for col in map_columns:
            if col in df.columns:
                plt.plot(df.index + 1, df[col], label=col.replace('metrics/', '').replace('val/', ''), linewidth=2)
        plt.title('mAP Progression', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('mAP Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'training_overview_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Detailed Loss Analysis
        plt.figure(figsize=(12, 8))
        
        # Create 2x2 subplot for detailed analysis
        loss_types = ['box_loss', 'cls_loss', 'dfl_loss']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for i, (loss_type, color) in enumerate(zip(loss_types, colors)):
            train_col = f'train/{loss_type}' if f'train/{loss_type}' in df.columns else loss_type
            val_col = f'val/{loss_type}' if f'val/{loss_type}' in df.columns else None
            
            plt.subplot(2, 2, i+1)
            if train_col in df.columns:
                plt.plot(df.index + 1, df[train_col], label=f'Train {loss_type}', color=color, linewidth=2)
            if val_col and val_col in df.columns:
                plt.plot(df.index + 1, df[val_col], label=f'Val {loss_type}', color=color, linestyle='--', linewidth=2)
            
            plt.title(f'{loss_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Overall metrics summary
        plt.subplot(2, 2, 4)
        final_metrics = df.iloc[-1]
        metrics_to_show = [col for col in df.columns if any(x in col for x in ['precision', 'recall', 'mAP50', 'mAP50-95'])]
        
        if metrics_to_show:
            values = [final_metrics[col] for col in metrics_to_show if col in final_metrics]
            labels = [col.replace('metrics/', '').replace('val/', '') for col in metrics_to_show]
            
            bars = plt.bar(range(len(values)), values, color=['#9b59b6', '#e67e22', '#1abc9c', '#34495e'])
            plt.title('Final Model Performance', fontsize=12, fontweight='bold')
            plt.ylabel('Score')
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'detailed_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("🎯 TRAINING SUMMARY")
        print("="*50)
        print(f"📊 Total Epochs: {len(df)}")
        print(f"🎯 Final mAP50: {final_metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"🎯 Final mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
        print(f"📈 Final Precision: {final_metrics.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"📈 Final Recall: {final_metrics.get('metrics/recall(B)', 'N/A'):.3f}")
        print(f"💾 Plots saved to: {output_dir}")
        print("="*50)
        
        logger.info(f"Training visualization completed. Plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        print(f"Available columns in results: {list(df.columns) if 'df' in locals() else 'Could not load data'}")

def create_model_comparison_plot(model_dir="models/train"):
    """Create a visual comparison of model performance."""
    try:
        results_path = Path(model_dir) / "results.csv"
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            return
            
        df = pd.read_csv(results_path)
        
        plt.figure(figsize=(10, 6))
        
        # Plot training progress
        epochs = df.index + 1
        plt.plot(epochs, df.get('metrics/mAP50(B)', [0]*len(df)), 
                label='mAP50', linewidth=3, color='#e74c3c')
        plt.plot(epochs, df.get('metrics/mAP50-95(B)', [0]*len(df)), 
                label='mAP50-95', linewidth=3, color='#3498db')
        
        plt.title('ShelfRanger Model Performance Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP Score', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add annotations for key milestones
        max_map50_idx = df['metrics/mAP50(B)'].idxmax() if 'metrics/mAP50(B)' in df.columns else 0
        max_map50_value = df.get('metrics/mAP50(B)', [0]).iloc[max_map50_idx] if 'metrics/mAP50(B)' in df.columns else 0
        
        plt.annotate(f'Best mAP50: {max_map50_value:.3f}', 
                    xy=(max_map50_idx + 1, max_map50_value), 
                    xytext=(max_map50_idx + 5, max_map50_value + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(Path(model_dir) / 'plots' / f'model_performance_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {str(e)}")

if __name__ == "__main__":
    # Generate plots from completed training
    print("🎨 Generating training visualization plots...")
    plot_training_results()
    create_model_comparison_plot()
    print("✅ Visualization complete!") 