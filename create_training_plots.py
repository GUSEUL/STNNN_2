"""
Create comprehensive training analysis plots from a model checkpoint.
Plots loss evolution (total, data, physics) and other specified metrics.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from data import MatDataset
from models import PhyCRNet
from matplotlib.gridspec import GridSpec

def create_comprehensive_plots(history, save_dir):
    """Creates and saves a comprehensive plot of training history."""
    
    if not history:
        print("History object is empty. Cannot generate plots.")
        return

    available_keys = history.keys()
    
    epochs = 0
    if 'epoch_train_loss' in available_keys:
        epochs = range(1, len(history['epoch_train_loss']) + 1)
    elif 'lr' in available_keys:
        epochs = range(1, len(history['lr']) + 1)
    
    if not epochs:
        print("Could not determine epoch count from history. Skipping plot generation.")
        return

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])  # Loss plot spanning two columns
    ax2 = fig.add_subplot(gs[1, 0])  # Data Loss
    ax3 = fig.add_subplot(gs[1, 1])  # Physics Loss
    ax4 = fig.add_subplot(gs[2, 0])  # Learning Rate
    ax5 = fig.add_subplot(gs[2, 1])  # Physics Weight

    if 'epoch_train_loss' in available_keys and 'epoch_val_loss' in available_keys:
        ax1.plot(epochs, history['epoch_train_loss'], 'b-o', markersize=3, label='Training Loss')
        ax1.plot(epochs, history['epoch_val_loss'], 'r-o', markersize=3, label='Validation Loss')
        ax1.set_title('Total Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')

    if 'epoch_train_data_loss' in available_keys:
        ax2.plot(epochs, history['epoch_train_data_loss'], 'g-o', markersize=3, label='Train Data Loss')
        if 'epoch_val_data_loss' in available_keys:
            ax2.plot(epochs, history['epoch_val_data_loss'], 'm-p', markersize=3, label='Val Data Loss')
        ax2.set_title('Data Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')

    if 'epoch_train_physics_loss' in available_keys:
        ax3.plot(epochs, history['epoch_train_physics_loss'], 'g-o', markersize=3, label='Train Physics Loss (Unweighted)')
        if 'epoch_val_physics_loss' in available_keys:
            ax3.plot(epochs, history['epoch_val_physics_loss'], 'm-p', markersize=3, label='Val Physics Loss (Unweighted)')
        ax3.set_title('Unweighted Physics Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')

    if 'lr' in available_keys:
        ax4.plot(epochs, history['lr'], 'm-o', markersize=3, label='Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True)

    if 'physics_weight' in available_keys:
        ax5.plot(epochs, history['physics_weight'], 'c-o', markersize=3, label='Physics Loss Weight')
        ax5.set_title('Physics Loss Weight')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('Weight')
        ax5.legend()
        ax5.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Comprehensive Training Analysis', fontsize=20)
    
    save_path = os.path.join(save_dir, 'comprehensive_training_analysis.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comprehensive training plot saved to {save_path}")

def create_error_metric_plots(history, save_dir):
    """Creates and saves plots for various error metrics vs. epoch."""
    if not history:
        print("History object is empty. Cannot generate error metric plots.")
        return

    available_keys = history.keys()
    
    epochs = 0
    if 'epoch_train_loss' in available_keys:
        epochs = range(1, len(history['epoch_train_loss']) + 1)
    
    if not epochs:
        print("Could not determine epoch count from history. Skipping error plot generation.")
        return

    fields = ['U', 'V', 'P', 'T']
    metrics = [
        ('abs_err', 'Mean Absolute Error'),
        ('mape', 'Mean Absolute Percentage Error (%)')
    ]
    
    # Check if any of the required keys are present
    has_any_metric = any(f'epoch_train_{m[0]}_{f}' in available_keys for f in fields for m in metrics)
    if not has_any_metric:
        print("No error metric data found in history. Skipping error metric plot generation.")
        return

    fig, axes = plt.subplots(len(fields), len(metrics), figsize=(20, 22))
    fig.suptitle('Error Metrics vs. Epoch', fontsize=20)

    for i, field in enumerate(fields):
        for j, (metric_key, metric_name) in enumerate(metrics):
            ax = axes[i, j]
            train_key = f'epoch_train_{metric_key}_{field}'
            val_key = f'epoch_val_{metric_key}_{field}'

            if train_key in available_keys:
                ax.plot(epochs, history[train_key], 'b-o', markersize=3, label='Training')
            if val_key in available_keys:
                ax.plot(epochs, history[val_key], 'r-o', markersize=3, label='Validation')
            
            ax.set_title(f'{metric_name} - {field}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric_name.split('(')[0].strip())
            ax.grid(True)
            ax.set_yscale('log')
            if train_key in available_keys or val_key in available_keys:
                ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(save_dir, 'error_metrics_vs_epoch.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Error metrics plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training plots from a model checkpoint.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the generated plots.')
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    history = checkpoint.get('history')
    
    if history:
        create_comprehensive_plots(history, args.save_dir)
        create_error_metric_plots(history, args.save_dir)
    else:
        print("No history found in the checkpoint.")

if __name__ == '__main__':
    main()