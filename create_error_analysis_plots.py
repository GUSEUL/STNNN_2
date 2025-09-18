import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import MatDataset
from models import PhyCRNet

def calculate_error_metrics(model, data_loader, device):
    """Calculates error metrics over the entire dataset."""
    model.eval()
    errors = {
        'U': {'max_abs': [], 'l2_rel': [], 'mape': []},
        'V': {'max_abs': [], 'l2_rel': [], 'mape': []},
        'P': {'max_abs': [], 'l2_rel': [], 'mape': []},
        'T': {'max_abs': [], 'l2_rel': [], 'mape': []},
    }
    field_names = ['U', 'V', 'P', 'T']
    
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for input_state, target_state, _ in tqdm(data_loader, desc="Gathering predictions"):
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            pred = model(input_state)
            
            all_targets.append(target_state.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # all_targets and all_preds are shape (Time, Channels, H, W)
    num_timesteps = all_targets.shape[0]

    for t in tqdm(range(num_timesteps), desc="Calculating errors per timestep"):
        for i, field in enumerate(field_names):
            target_t = all_targets[t, i, :, :]
            pred_t = all_preds[t, i, :, :]
            
            # Maximum Absolute Error
            max_abs_error = np.max(np.abs(target_t - pred_t))
            errors[field]['max_abs'].append(max_abs_error)
            
            # L2 Relative Error
            l2_norm_diff = np.linalg.norm(target_t - pred_t)
            l2_norm_target = np.linalg.norm(target_t)
            if l2_norm_target == 0:
                l2_relative_error = 0
            else:
                l2_relative_error = l2_norm_diff / l2_norm_target
            errors[field]['l2_rel'].append(l2_relative_error)

            # Mean Absolute Percentage Error (MAPE)
            epsilon = 1e-8
            mape = np.mean(np.abs((target_t - pred_t) / (target_t + epsilon))) * 100
            errors[field]['mape'].append(mape)
            
    return errors, num_timesteps

def plot_error_analysis(errors, num_timesteps, output_dir):
    """Plots the calculated error metrics."""
    field_names = list(errors.keys())
    timesteps = range(num_timesteps)

    fig, axes = plt.subplots(len(field_names), 3, figsize=(24, 6 * len(field_names)))
    fig.suptitle('Model Error Analysis over Time on Test Set', fontsize=20)

    for i, field in enumerate(field_names):
        # Plot Max Absolute Error
        ax1 = axes[i, 0]
        ax1.plot(timesteps, errors[field]['max_abs'], 'r-o', markersize=3)
        ax1.set_title(f'Maximum Absolute Error - {field}')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Max Abs Error')
        ax1.grid(True)
        ax1.set_yscale('log')

        # Plot L2 Relative Error
        ax2 = axes[i, 1]
        ax2.plot(timesteps, errors[field]['l2_rel'], 'b-o', markersize=3)
        ax2.set_title(f'L2 Relative Error - {field}')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('L2 Relative Error')
        ax2.grid(True)
        ax2.set_yscale('log')

        # Plot Mean Absolute Percentage Error
        ax3 = axes[i, 2]
        ax3.plot(timesteps, errors[field]['mape'], 'g-o', markersize=3)
        ax3.set_title(f'Mean Absolute Percentage Error - {field}')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('MAPE (%)')
        ax3.grid(True)
        ax3.set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(output_dir, 'error_analysis_plots.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Error analysis plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate error analysis plots from a trained model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .mat data file for evaluation.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated plots.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint_path}")
        return
        
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {'hidden': 128}) # Fallback for older models
    model = PhyCRNet(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")

    # Load test dataset
    full_dataset = MatDataset(args.data_path)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    
    test_indices = list(range(train_size + val_size, total_size))
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    print(f"Test dataset loaded with {len(test_dataset)} timesteps.")

    # Calculate and plot errors
    errors, num_timesteps = calculate_error_metrics(model, test_loader, device)
    plot_error_analysis(errors, num_timesteps, args.output_dir)

if __name__ == '__main__':
    main()
