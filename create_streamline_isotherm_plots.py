"""
Standalone script to generate streamline and isotherm plots, and animations
from a trained model checkpoint.
"""
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import MatDataset
from models import PhyCRNet
from enhanced_visualization import (
    create_flow_evolution_animation,
    create_streamline_plot,
    create_isotherm_plot
)

def generate_predictions(model, dataset, device, indices):
    """Generate predictions for a given set of indices."""
    model.eval()
    
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for i in tqdm(indices, desc="Generating Predictions"):
            input_state, target_state, _ = dataset[i]
            input_state = input_state.unsqueeze(0).to(device)
            
            prediction = model(input_state)
            
            all_targets.append(target_state.cpu().numpy())
            all_predictions.append(prediction.squeeze(0).cpu().numpy())
            
    return np.array(all_targets), np.array(all_predictions)

def plot_errors_from_history(history, output_dir):
    """Plots absolute and relative errors from the history object."""
    if not history or 'val_losses' not in history or not history['val_losses']:
        print("Warning: History object does not contain 'val_losses'. Skipping error plot.")
        return

    val_losses = history['val_losses']
    epochs = range(1, len(val_losses) + 1)

    # Absolute Error is the validation loss
    absolute_error = val_losses
    
    # Relative Error
    initial_loss = absolute_error[0] if absolute_error else 0
    relative_error = [(loss / initial_loss) * 100 if initial_loss > 0 else 0 for loss in absolute_error]

    plt.figure(figsize=(12, 5))

    # Absolute Error Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, absolute_error, label='Absolute Error (Validation Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Apply log scale to the y-axis

    # Relative Error Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, relative_error, label='Relative Error (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, 'error_vs_epoch.png')
    plt.savefig(error_plot_path, dpi=300)
    plt.close()
    print(f"Error vs. epoch plot saved to {error_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate streamline, isotherm, and animation visualizations.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .mat data file for evaluation.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the visualizations.')
    parser.add_argument('--create_animation', action='store_true', help='Flag to create GIF animations.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and history
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    history = checkpoint.get('history')
    
    model_config = checkpoint.get('model_config', {'hidden': 192})
    if 'input_channels' in model_config:
        model_config['ch'] = model_config.pop('input_channels')
    model = PhyCRNet(**model_config).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # Load dataset
    dataset = MatDataset(args.data_path)
    print(f"Dataset loaded from {args.data_path}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Plot errors from history if available
    if history:
        plot_errors_from_history(history, args.output_dir)
    else:
        print("Warning: No 'history' object found in checkpoint. Cannot generate error vs. epoch plot.")

    # Timesteps to visualize
    timestep_indices = [30, 50, 70]
    valid_indices = [min(max(0, t), len(dataset) - 1) for t in timestep_indices]

    targets, predictions = generate_predictions(model, dataset, device, valid_indices)

    for i, t_idx in enumerate(valid_indices):
        pred_U, pred_V, pred_T, _ = predictions[i]
        true_U, true_V, true_T, _ = targets[i]

        # Plot for Trained Model
        fig_pred_stream = create_streamline_plot(pred_U, pred_V, title=f'Trained Model Streamline at Timestep {t_idx}')
        fig_pred_stream.savefig(os.path.join(args.output_dir, f"timestep_{t_idx}_streamline_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_pred_stream)

        fig_pred_iso = create_isotherm_plot(pred_T, title=f'Trained Model Isotherm at Timestep {t_idx}')
        fig_pred_iso.savefig(os.path.join(args.output_dir, f"timestep_{t_idx}_isotherm_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_pred_iso)

        # Plot for Ground Truth
        fig_true_stream = create_streamline_plot(true_U, true_V, title=f'Ground Truth Streamline at Timestep {t_idx}')
        fig_true_stream.savefig(os.path.join(args.output_dir, f"timestep_{t_idx}_streamline_ground_truth.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_true_stream)

        fig_true_iso = create_isotherm_plot(true_T, title=f'Ground Truth Isotherm at Timestep {t_idx}')
        fig_true_iso.savefig(os.path.join(args.output_dir, f"timestep_{t_idx}_isotherm_ground_truth.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_true_iso)
        
        print(f"Visualizations for timestep {t_idx} saved.")

    # Animation (optional)
    if args.create_animation:
        print("Starting animation generation for all timesteps...")
        num_frames = len(dataset)
        animation_indices = list(range(num_frames))

        # We need to get the predictions for ALL timesteps
        _, predictions = generate_predictions(model, dataset, device, animation_indices)

        u_data = predictions[:, 0, :, :]
        v_data = predictions[:, 1, :, :]
        t_data = predictions[:, 2, :, :]

        animation_path = os.path.join(args.output_dir, 'flow_evolution_animation_all_steps.gif')
        create_flow_evolution_animation(
            u_data, v_data, t_data,
            animation_path,
            fps=15,  # Increased FPS for smoother animation
            title_prefix="Flow Evolution"
        )
        print(f"Full evolution animation saved to {animation_path}")

if __name__ == '__main__':
    main()
