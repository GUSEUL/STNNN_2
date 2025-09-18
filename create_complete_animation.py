"""
Create animation of the entire sequence of the dataset (target vs prediction).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse
from tqdm import tqdm
from data import MatDataset
from models import PhyCRNet

def generate_all_predictions(model, dataset, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Generating predictions for animation"):
            input_state, target_state, _ = dataset[i]
            input_state_gpu = input_state.unsqueeze(0).to(device)
            prediction = model(input_state_gpu)
            all_predictions.append(prediction.squeeze(0).cpu().numpy())
            all_targets.append(target_state.cpu().numpy())
    return np.array(all_targets), np.array(all_predictions)

def create_animation(targets, predictions, field_index, field_name, output_path, fps=15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{field_name} Field Evolution', fontsize=16)
    
    vmin = min(targets[:, field_index].min(), predictions[:, field_index].min())
    vmax = max(targets[:, field_index].max(), predictions[:, field_index].max())

    im1 = ax1.imshow(targets[0, field_index], cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title('Target')
    ax1.axis('off')

    im2 = ax2.imshow(predictions[0, field_index], cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title('Prediction')
    ax2.axis('off')
    
    fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.04)

    def update(frame):
        im1.set_data(targets[frame, field_index])
        im2.set_data(predictions[frame, field_index])
        fig.suptitle(f'{field_name} Field Evolution (Timestep {frame})', fontsize=16)
        return im1, im2

    anim = FuncAnimation(fig, update, frames=len(targets), blit=True)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create animations comparing target and prediction.")
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--data_path', required=True, type=str, help='Path to the .mat data file')
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save animations')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
        
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Instantiate model with the exact configuration from the checkpoint
    model_config = checkpoint.get('model_config', {'hidden': 192}) # Fallback for older models
    if 'input_channels' in model_config: # Handle old naming
        model_config['ch'] = model_config.pop('input_channels')
    model = PhyCRNet(**model_config).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")
    
    # Load Dataset
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    dataset = MatDataset(args.data_path)
    print(f"Using {len(dataset)} timesteps for animation.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    targets, predictions = generate_all_predictions(model, dataset, device)
    
    field_map = {0: 'U', 1: 'V', 2: 'T', 3: 'P'}
    for i, name in field_map.items():
        output_path = os.path.join(args.output_dir, f'animation_{name}.gif')
        create_animation(targets, predictions, i, name, output_path)
        
    print("All animations created successfully.")

if __name__ == '__main__':
    main() 