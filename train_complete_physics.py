"""
Complete Physics-Enhanced PhyCRNet Training
Uses the exact PDE system for natural convection with all physics terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import os
import glob
from data import MatDataset
from models import PhyCRNet
from accurate_physics_loss import AccuratePhysicsLoss

def calculate_error_metrics(prediction, target_state):
    """Calculates absolute error and MAPE for each field."""
    metrics = {}
    epsilon = 1e-8
    fields = ['U', 'V', 'P', 'T']
    
    for i, field in enumerate(fields):
        pred_field = prediction[:, i, :, :]
        target_field = target_state[:, i, :, :]
        
        # Mean Absolute Error (L1 Loss)
        metrics[f'abs_err_{field}'] = F.l1_loss(pred_field, target_field).item()
        
        # Mean Absolute Percentage Error
        mape_tensor = torch.abs((target_field - pred_field) / (target_field.abs() + epsilon)) * 100
        metrics[f'mape_{field}'] = mape_tensor.mean().item()
    
    

    return metrics

def train_complete_physics_model():
    """Train PhyCRNet with the complete PDE system implementation."""
    
    print("Complete Physics-Enhanced PhyCRNet Training")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get parameters from environment variables
    data_dir = os.environ.get('DATA_DIR', 'data')
    data_file = os.environ.get('DATA_FILE') # Get specific data file if provided
    save_dir = os.environ.get('SAVE_DIR', 'results')
    num_epochs = int(os.environ.get('EPOCHS', '200'))
    batch_size = int(os.environ.get('BATCH_SIZE', '32'))
    learning_rate = float(os.environ.get('LEARNING_RATE', '0.001'))
    num_files = int(os.environ.get('NUM_FILES', '1'))
    time_slice_end = os.environ.get('TIME_SLICE_END')

    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate, 
        'save_interval': 50,
        'model_save_path': os.path.join(save_dir, 'complete_physics_model_checkpoint.pth'),
        'results_dir': save_dir,
        'data_dir': data_dir,
        'num_files': num_files,
        'physics_weight_initial': 0.01,
        'physics_weight_max': 0.1,
        'data_weight': 1.0,
        'warmup_epochs': 10,
        'fraction_train': 0.7,
        'fraction_val': 0.2,
        'fraction_test': 0.1,
        'grad_clip_max_norm': 1.0,  # More standard gradient clipping value
        'scheduler_patience': 15,    # More aggressive patience
        'scheduler_factor': 0.5,    # More aggressive factor
    }
    
    print(f"Device: {device}")
    print(f"Learning Rate: {config['learning_rate']:.1e}")
    print(f"LR Scheduler: ReduceLROnPlateau(patience={config['scheduler_patience']}, factor={config['scheduler_factor']})")
    print(f"Gradient Clipping: max_norm={config['grad_clip_max_norm']}")
    if time_slice_end:
        print(f"Time Slice End: {time_slice_end}s")

    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Dataset Loading Logic
    if data_file:
        if not os.path.exists(data_file):
            raise ValueError(f"Specified data file not found: {data_file}")
        mat_files = [data_file]
        print(f"Loading specified data file: {data_file}")
    else:
        data_pattern = os.path.join(config['data_dir'], '*.mat')
        mat_files = sorted(glob.glob(data_pattern))[:config['num_files']]
        print(f"Loading {len(mat_files)} files from directory: {config['data_dir']}")

    if not mat_files:
        raise ValueError(f"No .mat files found in {config['data_dir']} or specified via --data_file")

    dataset_kwargs = {'device': device}
    if time_slice_end:
        dataset_kwargs['time_slice_end'] = float(time_slice_end)
    
    dataset = MatDataset(mat_files[0], **dataset_kwargs)
    dataset_params = dataset.get_params()
    nanofluid_props = dataset.get_nanofluid_properties()
    
    # Data Split
    total_size = len(dataset)
    train_size = int(config['fraction_train'] * total_size)
    val_size = int(config['fraction_val'] * total_size)
    
    # Ensure val_size is at least 1 if total_size is small
    if total_size > train_size and val_size == 0:
        val_size = 1

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    print(f"\nDataset size: {total_size} -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    if not train_indices:
        raise ValueError("Training set is empty. Check data or time_slice_end.")
    if not val_indices:
        print("Warning: Validation set is empty. Using last training sample for validation.")
        val_indices = [train_indices[-1]]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    model_config = {'ch': 4, 'hidden': 128, 'upscale': 1, 'dropout_rate': 0.1}
    model = PhyCRNet(**model_config).to(device)
    
    # Physics Loss
    physics_loss_fn = AccuratePhysicsLoss(
        params=dataset_params,
        nanofluid_props=nanofluid_props,
        dt=dataset_params.get('dt', 0.0001)
    ).to(device)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=config['scheduler_patience'], 
        factor=config['scheduler_factor']
    )
    
    # Training History
    history = {
        'epoch_train_loss': [], 'epoch_val_loss': [],
        'epoch_train_data_loss': [], 'epoch_val_data_loss': [],
        'epoch_train_physics_loss': [], 'epoch_val_physics_loss': [],
        'lr': [], 'physics_weight': [],
    }
    fields = ['U', 'V', 'P', 'T']
    for field in fields:
        for metric in ['abs_err', 'mape']:
            history[f'epoch_train_{metric}_{field}'] = []
            history[f'epoch_val_{metric}_{field}'] = []

    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 80)
    
    total_training_start = time.time()
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        
        epoch_stats = {'train_loss': 0.0, 'data_loss': 0.0, 'physics_loss': 0.0, 'batches': 0}
        for field in fields:
            for metric in ['abs_err', 'mape']:
                epoch_stats[f'train_{metric}_{field}'] = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}')
        for input_state, target_state, _ in progress_bar:
            input_state, target_state = input_state.to(device), target_state.to(device)
            
            optimizer.zero_grad()
            
            prediction = model(input_state)
            
            loss_data = F.mse_loss(prediction, target_state)
            
            physics_result = physics_loss_fn(input_state, prediction, validation_mode=True)
            loss_physics = physics_result['total']
            
            physics_weight = config['physics_weight_initial'] # Simplified weight for now
            
            total_loss = config['data_weight'] * loss_data + physics_weight * loss_physics
            
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_max_norm'])
                optimizer.step()
                
                epoch_stats['train_loss'] += total_loss.item()
                epoch_stats['data_loss'] += loss_data.item()
                epoch_stats['physics_loss'] += loss_physics.item()
                
                with torch.no_grad():
                    batch_metrics = calculate_error_metrics(prediction, target_state)
                    for key, value in batch_metrics.items():
                        epoch_stats[f'train_{key}'] += value

                epoch_stats['batches'] += 1
            
            progress_bar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        # Average training stats
        if epoch_stats['batches'] > 0:
            for key in epoch_stats:
                if key != 'batches':
                    epoch_stats[key] /= epoch_stats['batches']

        # --- Validation Phase ---
        model.eval()
        val_stats = {'val_loss': 0.0, 'data_loss': 0.0, 'physics_loss': 0.0, 'batches': 0}
        for field in fields:
            for metric in ['abs_err', 'mape']:
                val_stats[f'val_{metric}_{field}'] = 0.0

        with torch.no_grad():
            for input_state, target_state, _ in val_loader:
                input_state, target_state = input_state.to(device), target_state.to(device)
                prediction = model(input_state)
                
                loss_data = F.mse_loss(prediction, target_state)
                physics_result = physics_loss_fn(input_state, prediction, validation_mode=True)
                loss_physics = physics_result['total']
                
                total_loss = config['data_weight'] * loss_data + physics_weight * loss_physics
                
                val_stats['val_loss'] += total_loss.item()
                val_stats['data_loss'] += loss_data.item()
                val_stats['physics_loss'] += loss_physics.item()
                
                batch_metrics = calculate_error_metrics(prediction, target_state)
                for key, value in batch_metrics.items():
                    val_stats[f'val_{key}'] += value
                
                val_stats['batches'] += 1
        
        # Average validation stats
        if val_stats['batches'] > 0:
            for key in val_stats:
                if key != 'batches':
                    val_stats[key] /= val_stats['batches']
        
        # Update scheduler
        scheduler.step(val_stats['val_loss'])
        
        # Store history
        history['epoch_train_loss'].append(epoch_stats['train_loss'])
        history['epoch_val_loss'].append(val_stats['val_loss'])
        history['epoch_train_data_loss'].append(epoch_stats['data_loss'])
        history['epoch_val_data_loss'].append(val_stats['data_loss'])
        history['epoch_train_physics_loss'].append(epoch_stats['physics_loss'])
        history['epoch_val_physics_loss'].append(val_stats['physics_loss'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['physics_weight'].append(physics_weight)
        for field in fields:
            for metric in ['abs_err', 'mape']:
                history[f'epoch_train_{metric}_{field}'].append(epoch_stats[f'train_{metric}_{field}'])
                history[f'epoch_val_{metric}_{field}'].append(val_stats[f'val_{metric}_{field}'])

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} ({epoch_time:.1f}s) | Train Loss: {epoch_stats['train_loss']:.6f} | Val Loss: {val_stats['val_loss']:.6f}")

        if val_stats['val_loss'] < best_val_loss:
            best_val_loss = val_stats['val_loss']
            torch.save({
                'epoch': epoch,
                'model_config': model_config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'config': config,
                'dataset_params': dataset_params
            }, config['model_save_path'])
            print(f"  Best model saved! Val loss: {best_val_loss:.6f}")

    total_training_time = time.time() - total_training_start
    print(f"\nTRAINING COMPLETED! Total time: {total_training_time/3600:.2f} hours")
    
    return config['model_save_path']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PhyCRNet with complete physics')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--data_file', type=str, default=None, help='Path to a specific data file to train on.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_files', type=int, default=1, help='Number of files to use for training')
    parser.add_argument('--time_slice_end', type=float, default=None, help='End time to slice the dataset (e.g., 0.5, 1.0)')

    args = parser.parse_args()
    
    # Set environment variables for the training function
    os.environ['DATA_DIR'] = args.data_dir
    os.environ['SAVE_DIR'] = args.save_dir
    os.environ['EPOCHS'] = str(args.epochs)
    os.environ['BATCH_SIZE'] = str(args.batch_size)
    os.environ['LEARNING_RATE'] = str(args.learning_rate)
    os.environ['NUM_FILES'] = str(args.num_files)
    if args.data_file:
        os.environ['DATA_FILE'] = args.data_file
    if args.time_slice_end is not None:
        os.environ['TIME_SLICE_END'] = str(args.time_slice_end)
    
    train_complete_physics_model()