"""
Complete Physics-Enhanced PhyCRNet Training
Uses the exact PDE system for natural convection with all physics terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from datetime import datetime
from data import MatDataset
from models import PhyCRNet
from accurate_physics_loss import AccuratePhysicsLoss

def train_complete_physics_model():
    """Train PhyCRNet with the complete PDE system implementation."""
    
    print("Complete Physics-Enhanced PhyCRNet Training")
    print("Implementing Full PDE System with Nanofluid Properties:")
    print("   1. Continuity: ∂U/∂X + ∂V/∂Y = 0")
    print("   2. X-momentum: ∂U/∂t + U∂U/∂X + V∂U/∂Y = -∂P/∂X + (ν_thnf/ν_f)Pr[∇²U] - (ν_thnf/ν_f)(Pr/Da)U")
    print("   3. Y-momentum: ∂V/∂t + U∂V/∂X + V∂V/∂Y = -∂P/∂Y + (ν_thnf/ν_f)Pr[∇²V] + (β_thnf/β_f)Ra·Pr·θ")
    print("              - (ν_thnf/ν_f)(Pr/Da)V - (σ_thnf/σ_f)(ρ_f/ρ_thnf)Ha²·Pr·V")
    print("   4. Energy: ∂θ/∂t + U∂θ/∂X + V∂θ/∂Y = (α_thnf/α_f)[∇²θ] + (ρC_p)_f/(ρC_p)_thnf Q·θ")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'num_epochs': 500,
        'batch_size': 32,
        'learning_rate': 1e-3, 
        'save_interval': 50,
        'data_file': '01_Da_0.100.mat',
        'model_save_path': 'complete_physics_model_Da_0.100_checkpoint.pth',
        'physics_weight_initial': 0.01, 
        'physics_weight_max': 0.1,    
        'data_weight': 1.0,
        'warmup_epochs': 20,
        'parameter_weight': 0.001,
        'fraction_train': 0.7,  # 70% for training
        'fraction_val': 0.2,    # 20% for validation
        'fraction_test': 0.1,   # 10% for testing
        # Dynamic weight adjustment parameters
        'dynamic_weights': True,          # Enable dynamic weight adjustment
        'weight_adjust_interval': 10,     # Adjust weights every N batches
        'weight_adjust_warmup': 150,      # Start adjusting after N epochs
        'min_ratio_threshold': 0.2,       # Minimum ratio before adjustment (5x difference)
        'max_boost_factor': 11.0,         # Maximum factor to boost any weight
    }
    
    print(f"Device: {device}")
    print(f"Physics weight: {config['physics_weight_initial']} → {config['physics_weight_max']}")
    print(f"Dynamic weight adjustment: {'Enabled' if config['dynamic_weights'] else 'Disabled'}")
    if config['dynamic_weights']:
        print(f"  Adjustment parameters:")
        print(f"    - Warmup epochs: {config['weight_adjust_warmup']}")
        print(f"    - Adjustment interval: every {config['weight_adjust_interval']} batches")
        print(f"    - Ratio threshold: {config['min_ratio_threshold']} (1/{1/config['min_ratio_threshold']:.0f} = {1/config['min_ratio_threshold']:.0f}x)")
        print(f"    - Max boost factor: {config['max_boost_factor']}x")
    
    # Create directories
    os.makedirs('complete_physics_results', exist_ok=True)
    
    # Dataset (automatically loads nanofluid properties)
    dataset = MatDataset(config['data_file'], device=device)
    dataset_params = dataset.get_params()
    nanofluid_props = dataset.get_nanofluid_properties()
    
    # Physics parameters and nanofluid properties are loaded from MATLAB file
    
    print(f"Physical Parameters:")
    print(f"   Ra = {dataset_params['Ra']:.1e} (Rayleigh)")
    print(f"   Pr = {dataset_params['Pr']:.3f} (Prandtl)")
    print(f"   Ha = {dataset_params['Ha']:.1f} (Hartmann)")
    print(f"   Da = {dataset_params['Da']:.1e} (Darcy)")
    print(f"   Rd = {dataset_params['Rd']:.2f} (Radiation)")
    print(f"   Q = {dataset_params['Q']:.3f} (Heat source)")
    
    # Split dataset by time step (not randomly) into train/val/test
    # Use early time steps for training, middle for validation, later for testing
    total_size = len(dataset)
    train_size = int(config['fraction_train'] * total_size)
    val_size = int(config['fraction_val'] * total_size)
    test_size = total_size - train_size - val_size  # Remaining goes to test
    
    print(f"\nDataset Split Information:")
    print(f"  Total timesteps: {total_size}")
    print(f"  Train: {train_size} timesteps ({config['fraction_train']*100:.1f}%) - indices 0 to {train_size-1}")
    print(f"  Validation: {val_size} timesteps ({config['fraction_val']*100:.1f}%) - indices {train_size} to {train_size+val_size-1}")
    print(f"  Test: {test_size} timesteps ({test_size/total_size*100:.1f}%) - indices {train_size+val_size} to {total_size-1}")
    
    # Create indices for time-based split
    train_indices = list(range(0, train_size))  # Early time steps
    val_indices = list(range(train_size, train_size + val_size))  # Middle time steps
    test_indices = list(range(train_size + val_size, total_size))  # Later time steps
    
    # Verify split integrity
    assert len(train_indices) == train_size, f"Train indices mismatch: {len(train_indices)} != {train_size}"
    assert len(val_indices) == val_size, f"Val indices mismatch: {len(val_indices)} != {val_size}"
    assert len(test_indices) == test_size, f"Test indices mismatch: {len(test_indices)} != {test_size}"
    assert len(set(train_indices + val_indices + test_indices)) == total_size, "Overlapping indices detected!"
    print("  ✓ Dataset split verification passed - no overlaps detected")
    
    # Create subsets based on time indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model with Ra prediction enabled
    model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1, predict_ra=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters (with Ra prediction)")
    print(f"Target Ra value: {dataset_params['Ra']:.1e}")
    
    # Physics loss configuration
    use_predicted_ra = True  # Set to True to use predicted Ra in physics equations
    enable_physics_loss = True  # Set to True to enable physics loss
    
    physics_loss = AccuratePhysicsLoss(
        params=dataset_params,
        nanofluid_props=nanofluid_props,  # Use nanofluid properties from data.py
        dt=dataset_params.get('dt', 0.0001),
        dx=1.0, dy=1.0,
        use_predicted_ra=use_predicted_ra
    ).to(device) if enable_physics_loss else None
    
    print(f"Physics loss enabled: {enable_physics_loss}")
    if not enable_physics_loss:
        print("Running in data-only mode for stability")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.8)
    
    # Training tracking
    history = {
        'train_losses': [], 'val_losses': [], 'physics_losses': [], 'data_losses': [],
        'continuity': [], 'momentum_x': [], 'momentum_y': [], 'energy': [],
        'unweighted_physics': [], 'predicted_ra': [], 'ra_loss': []
    }
    best_val_loss = float('inf')
    worst_val_loss = 0.0
    worst_val_ra = 0.0
    
    print("Starting training...")
    print("=" * 80)
    
    # Initialize timing
    total_training_start = time.time()
    epoch_times = []
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        if physics_loss is not None:
            physics_loss.set_epoch(epoch)
        
        # Progressive physics weight
        if epoch < config['warmup_epochs']:
            progress = epoch / config['warmup_epochs']
            physics_weight = config['physics_weight_initial'] * progress
        else:
            progress = min((epoch - config['warmup_epochs']) / 30.0, 1.0)
            physics_weight = (config['physics_weight_initial'] + 
                            (config['physics_weight_max'] - config['physics_weight_initial']) * progress)
        
        # Training
        model.train()
        epoch_stats = {
            'train_loss': 0.0, 'physics_loss': 0.0, 'data_loss': 0.0, 'ra_loss': 0.0,
            'continuity': 0.0, 'momentum_x': 0.0, 'momentum_y': 0.0, 'energy': 0.0,
            'unweighted_physics': 0.0, 'predicted_ra': 0.0, 'batches': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}')
        for batch_idx, (input_state, target_state, _) in enumerate(progress_bar):
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            prediction, ra_scalar = model(input_state)  # Get both prediction and Ra scalar
            
            # Create extended target with Ra values
            batch_size = target_state.size(0)
            target_ra = torch.full((batch_size, 1), dataset_params['Ra'], 
                                 device=device, dtype=target_state.dtype)
            
            # Extend target_state to include Ra channel
            target_ra_spatial = target_ra.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *target_state.shape[2:])
            target_extended = torch.cat([target_state, target_ra_spatial], dim=1)
            
            # Data loss (for main fields U, V, T, P)
            loss_data_main = F.mse_loss(prediction[:, :4], target_state)
            
            # Ra prediction loss - model now outputs log10(Ra) directly
            target_ra_log = torch.log10(target_ra)  # Convert target to log scale
            loss_ra = F.mse_loss(ra_scalar, target_ra_log)  # ra_scalar is already log10(Ra)
            
            # Combined data loss
            loss_data = loss_data_main + config['parameter_weight'] * loss_ra  # Weight Ra loss lower
            
            # Physics loss with components (if enabled)
            physics_result = None
            loss_physics_total = torch.tensor(0.0, device=device)
            total_loss = config['data_weight'] * loss_data  # Default to data loss only
            
            if enable_physics_loss and physics_loss is not None:
                try:
                    if use_predicted_ra:
                        physics_result = physics_loss(input_state, prediction[:, :4], validation_mode=True, ra_scalar=ra_scalar)
                    else:
                        physics_result = physics_loss(input_state, prediction[:, :4], validation_mode=True)
                        
                    if physics_result is not None and isinstance(physics_result, dict) and 'total' in physics_result:
                        loss_physics_total = physics_result['total']
                        # Ensure physics loss is a scalar tensor
                        if hasattr(loss_physics_total, 'dim') and loss_physics_total.dim() > 0:
                            loss_physics_total = loss_physics_total.mean()
                        
                        # Dynamic weight adjustment: check if temperature (energy) loss is significantly larger
                        if (config['dynamic_weights'] and 
                            epoch > config['weight_adjust_warmup'] and 
                            batch_idx % config['weight_adjust_interval'] == 0):
                            try:
                                weight_adjusted = physics_loss.adjust_weights_dynamically(
                                    physics_result, 
                                    min_ratio_threshold=config['min_ratio_threshold'],
                                    max_boost_factor=config['max_boost_factor']
                                )
                                if weight_adjusted:
                                    # Recalculate physics loss with new weights
                                    if use_predicted_ra:
                                        physics_result = physics_loss(input_state, prediction[:, :4], validation_mode=True, ra_scalar=ra_scalar)
                                    else:
                                        physics_result = physics_loss(input_state, prediction[:, :4], validation_mode=True)
                                    loss_physics_total = physics_result['total']
                                    if hasattr(loss_physics_total, 'dim') and loss_physics_total.dim() > 0:
                                        loss_physics_total = loss_physics_total.mean()
                            except Exception as e:
                                print(f"Warning - Dynamic weight adjustment failed: {e}")
                        
                        # Add physics loss to total
                        total_loss = config['data_weight'] * loss_data + physics_weight * loss_physics_total
                    else:
                        # Physics result is invalid
                        raise ValueError(f"Physics result is invalid: {type(physics_result)}")
                        
                except Exception as e:
                    print(f"Warning - Physics error: {e}")
                    # Fall back to data-only loss
                    total_loss = config['data_weight'] * loss_data
                    physics_result = None
                    loss_physics_total = torch.tensor(0.0, device=device)
            
            # Backward pass
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                # Update statistics
                epoch_stats['train_loss'] += total_loss.item()
                epoch_stats['physics_loss'] += loss_physics_total.item()
                epoch_stats['data_loss'] += loss_data.item()
                epoch_stats['ra_loss'] += loss_ra.item()
                
                # Add physics component stats if available (store both weighted and unweighted)
                if physics_result is not None and isinstance(physics_result, dict):
                    try:
                        # Get unweighted physics loss for comparison
                        unweighted_total = 0.0
                        for key in ['continuity', 'momentum_x', 'momentum_y', 'energy']:
                            if key in physics_result:
                                value = physics_result[key]
                                if hasattr(value, 'dim') and value.dim() == 0:
                                    epoch_stats[key] += value.item()
                                    # Calculate unweighted contribution
                                    if key == 'continuity':
                                        unweighted_total += value.item() / physics_loss.w_continuity
                                    elif key == 'momentum_x':
                                        unweighted_total += value.item() / physics_loss.w_momentum_x
                                    elif key == 'momentum_y':
                                        unweighted_total += value.item() / physics_loss.w_momentum_y
                                    elif key == 'energy':
                                        unweighted_total += value.item() / physics_loss.w_energy
                                elif hasattr(value, 'mean'):
                                    epoch_stats[key] += value.mean().item()
                                    # Calculate unweighted contribution
                                    mean_val = value.mean().item()
                                    if key == 'continuity':
                                        unweighted_total += mean_val / physics_loss.w_continuity
                                    elif key == 'momentum_x':
                                        unweighted_total += mean_val / physics_loss.w_momentum_x
                                    elif key == 'momentum_y':
                                        unweighted_total += mean_val / physics_loss.w_momentum_y
                                    elif key == 'energy':
                                        unweighted_total += mean_val / physics_loss.w_energy
                                else:
                                    epoch_stats[key] += float(value)
                                    if key == 'continuity':
                                        unweighted_total += float(value) / physics_loss.w_continuity
                                    elif key == 'momentum_x':
                                        unweighted_total += float(value) / physics_loss.w_momentum_x
                                    elif key == 'momentum_y':
                                        unweighted_total += float(value) / physics_loss.w_momentum_y
                                    elif key == 'energy':
                                        unweighted_total += float(value) / physics_loss.w_energy
                        
                        # Store unweighted physics loss
                        if 'unweighted_physics' not in epoch_stats:
                            epoch_stats['unweighted_physics'] = 0.0
                        epoch_stats['unweighted_physics'] += unweighted_total
                        
                    except Exception as e:
                        print(f"Warning - Error extracting physics components: {e}")
                
                epoch_stats['predicted_ra'] += torch.pow(10, ra_scalar.mean()).item()  # Convert log10(Ra) back to Ra
                epoch_stats['batches'] += 1
            else:
                print(f"Warning - Invalid total loss (NaN or Inf): {total_loss}")
                # Use data loss as fallback
                loss_data.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_stats['train_loss'] += loss_data.item()
                epoch_stats['data_loss'] += loss_data.item()
                epoch_stats['ra_loss'] += loss_ra.item()
                epoch_stats['predicted_ra'] += torch.pow(10, ra_scalar.mean()).item()  # Convert log10(Ra) back to Ra
                epoch_stats['batches'] += 1
            
            # Update progress bar
            current_loss = total_loss.item() if 'total_loss' in locals() and total_loss is not None else loss_data.item()
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Physics': f'{physics_weight:.4f}',
                'Ra': f'{torch.pow(10, ra_scalar.mean()).item():.1e}'  # Convert log10(Ra) to Ra for display
            })
        
        # Average training statistics
        if epoch_stats['batches'] > 0:
            for key in epoch_stats:
                if key != 'batches':
                    epoch_stats[key] /= epoch_stats['batches']
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ra_avg = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for input_state, target_state, _ in val_loader:
                input_state = input_state.to(device)
                target_state = target_state.to(device)
                prediction, ra_scalar = model(input_state)
                
                # Validation loss (only main fields)
                loss = F.mse_loss(prediction[:, :4], target_state)
                val_loss += loss.item()
                val_ra_avg += torch.pow(10, ra_scalar.mean()).item()  # Convert log10(Ra) back to Ra
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        avg_val_ra = val_ra_avg / val_batches if val_batches > 0 else 0.0
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_losses'].append(epoch_stats['train_loss'])
        history['val_losses'].append(avg_val_loss)
        history['physics_losses'].append(epoch_stats['physics_loss'])
        history['data_losses'].append(epoch_stats['data_loss'])
        history['continuity'].append(epoch_stats['continuity'])
        history['momentum_x'].append(epoch_stats['momentum_x'])
        history['momentum_y'].append(epoch_stats['momentum_y'])
        history['energy'].append(epoch_stats['energy'])
        history['unweighted_physics'].append(epoch_stats['unweighted_physics'])
        history['predicted_ra'].append(epoch_stats['predicted_ra'])
        history['ra_loss'].append(epoch_stats['ra_loss'])
        
        # Calculate timing
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        total_elapsed = time.time() - total_training_start
        avg_epoch_time = np.mean(epoch_times)
        estimated_remaining = avg_epoch_time * (config['num_epochs'] - epoch - 1)
        
        # Print progress with timing information
        print(f"\nEpoch {epoch+1:3d}/{config['num_epochs']} ({epoch_time:.1f}s | Avg: {avg_epoch_time:.1f}s | Total: {total_elapsed/60:.1f}m | ETA: {estimated_remaining/60:.1f}m)")
        print(f"  Train: {epoch_stats['train_loss']:.6f} (Data: {epoch_stats['data_loss']:.6f}, Physics: {epoch_stats['physics_loss']:.6f} [Raw: {epoch_stats['unweighted_physics']:.6f}], Ra: {epoch_stats['ra_loss']:.6f})")
        print(f"  Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}, Physics Weight: {physics_weight:.6f}")
        print(f"  Predicted Ra: {epoch_stats['predicted_ra']:.6f} (Target: {dataset_params['Ra']:.6f}, Val: {avg_val_ra:.6f})")
        
        if epoch % 10 == 0:
            print(f"  Physics Components (with weights):")
            print(f"     Continuity: {epoch_stats['continuity']:.6f} (w={physics_loss.w_continuity:.1f})")
            print(f"     Momentum X: {epoch_stats['momentum_x']:.6f} (w={physics_loss.w_momentum_x:.1f})")
            print(f"     Momentum Y: {epoch_stats['momentum_y']:.6f} (w={physics_loss.w_momentum_y:.1f})")
            print(f"     Energy: {epoch_stats['energy']:.6f} (w={physics_loss.w_energy:.1f})")
        
        # Track worst validation loss and corresponding Ra
        if avg_val_loss > worst_val_loss:
            worst_val_loss = avg_val_loss
            worst_val_ra = avg_val_ra
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'worst_val_loss': worst_val_loss,
                'worst_val_ra': worst_val_ra,
                'config': config,
                'dataset_params': dataset_params
            }, config['model_save_path'])
            print(f"  Best model saved! Val loss: {best_val_loss:.6f}")
        
        # Save visualization and epoch analysis
        if (epoch + 1) % config['save_interval'] == 0:
            print(f"  Saving epoch {epoch+1} visualization and analysis...")
            save_complete_visualization(model, val_loader, history, epoch+1, device, dataset_params)
            
            # Save epoch analysis (target vs prediction vs error comparison)
            saved_analysis = save_epoch_analysis(model, dataset, epoch+1, device, 'complete_physics_results')
            if saved_analysis:
                print(f"  Epoch analysis complete: {len(saved_analysis)} comparison figures saved")
    
    # Calculate final timing statistics
    total_training_time = time.time() - total_training_start
    avg_epoch_time = np.mean(epoch_times)
    
    print(f"\n" + "=" * 80)
    print(f"TRAINING COMPLETED!")
    print(f"=" * 80)
    print(f"Training Time Summary:")
    print(f"  Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"  Average epoch time: {avg_epoch_time:.1f} seconds")
    print(f"  Fastest epoch: {min(epoch_times):.1f} seconds")
    print(f"  Slowest epoch: {max(epoch_times):.1f} seconds")
    print(f"\nModel Performance:")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Final predicted Ra: {history['predicted_ra'][-1]:.1e} (Target: {dataset_params['Ra']:.1e})")
    
    # Test set evaluation
    print(f"\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_ra_avg = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for input_state, target_state, _ in test_loader:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            prediction, ra_scalar = model(input_state)
            
            # Test loss (only main fields)
            loss = F.mse_loss(prediction[:, :4], target_state)
            test_loss += loss.item()
            test_ra_avg += torch.pow(10, ra_scalar.mean()).item()  # Convert log10(Ra) back to Ra
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches if test_batches > 0 else float('inf')
    avg_test_ra = test_ra_avg / test_batches if test_batches > 0 else 0.0
    
    print(f"Test Results:")
    print(f"  Test loss: {avg_test_loss:.6f}")
    print(f"  Test predicted Ra: {avg_test_ra:.1f} (Target: {dataset_params['Ra']:.1f})")
    ra_error_abs = abs(avg_test_ra - dataset_params['Ra'])
    ra_error_rel = (ra_error_abs / dataset_params['Ra']) * 100
    print(f"  Ra prediction error: {ra_error_abs:.1f} (Relative: {ra_error_rel:.3f}%)")
    
    return config['model_save_path']

def save_complete_visualization(model, val_loader, history, epoch, device, dataset_params=None):
    """Save comprehensive visualization."""
    
    # Get sample prediction
    model.eval()
    predicted_ra_values = []
    with torch.no_grad():
        for input_state, target_state, _ in val_loader:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            prediction, ra_scalar = model(input_state)
            
            input_img = input_state[0].cpu().numpy()
            target_img = target_state[0].cpu().numpy()
            pred_img = prediction[0, :4].cpu().numpy()  # Only main fields for visualization
            predicted_ra_values.append(ra_scalar.cpu().numpy())
            break
    
    # Calculate Ra statistics - convert from log10(Ra) to Ra
    avg_predicted_ra = float(np.power(10, np.mean(predicted_ra_values[0])))
    
    # Loss analysis - expand to include Ra tracking
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # type: ignore
    
    # Total losses
    axes[0, 0].plot(history['train_losses'], label='Train', linewidth=2)  # type: ignore
    axes[0, 0].plot(history['val_losses'], label='Validation', linewidth=2)  # type: ignore
    axes[0, 0].set_title('Total Loss')  # type: ignore
    axes[0, 0].legend()  # type: ignore
    axes[0, 0].grid(True, alpha=0.3)  # type: ignore
    
    # Component losses
    axes[0, 1].plot(history['data_losses'], label='Data', linewidth=2)  # type: ignore
    axes[0, 1].plot(history['physics_losses'], label='Physics', linewidth=2)  # type: ignore
    axes[0, 1].set_title('Component Losses')  # type: ignore
    axes[0, 1].legend()  # type: ignore
    axes[0, 1].grid(True, alpha=0.3)  # type: ignore
    
    # Physics terms
    axes[0, 2].plot(history['continuity'], label='Continuity', linewidth=2)  # type: ignore
    axes[0, 2].plot(history['momentum_x'], label='Momentum X', linewidth=2)  # type: ignore
    axes[0, 2].plot(history['momentum_y'], label='Momentum Y', linewidth=2)  # type: ignore
    axes[0, 2].plot(history['energy'], label='Energy', linewidth=2)  # type: ignore
    axes[0, 2].set_title('Physics Terms')  # type: ignore
    axes[0, 2].legend()  # type: ignore
    axes[0, 2].grid(True, alpha=0.3)  # type: ignore
    
    # Log scale versions
    axes[1, 0].semilogy(history['train_losses'], label='Train', linewidth=2)  # type: ignore
    axes[1, 0].semilogy(history['val_losses'], label='Validation', linewidth=2)  # type: ignore
    axes[1, 0].set_title('Total Loss (Log)')  # type: ignore
    axes[1, 0].legend()  # type: ignore
    axes[1, 0].grid(True, alpha=0.3)  # type: ignore
    
    axes[1, 1].semilogy(history['data_losses'], label='Data', linewidth=2)  # type: ignore
    axes[1, 1].semilogy([max(x, 1e-10) for x in history['physics_losses']], label='Physics', linewidth=2)  # type: ignore
    axes[1, 1].set_title('Component Losses (Log)')  # type: ignore
    axes[1, 1].legend()  # type: ignore
    axes[1, 1].grid(True, alpha=0.3)  # type: ignore
    
    # Recent physics terms
    recent = max(1, len(history['continuity']) - 20)
    axes[1, 2].plot(range(recent, len(history['continuity'])), history['continuity'][recent:],   # type: ignore
                   label='Continuity', linewidth=2)
    axes[1, 2].plot(range(recent, len(history['momentum_x'])), history['momentum_x'][recent:],   # type: ignore
                   label='Momentum X', linewidth=2)
    axes[1, 2].plot(range(recent, len(history['momentum_y'])), history['momentum_y'][recent:],   # type: ignore
                   label='Momentum Y', linewidth=2)
    axes[1, 2].plot(range(recent, len(history['energy'])), history['energy'][recent:],   # type: ignore
                   label='Energy', linewidth=2)
    axes[1, 2].set_title('Recent Physics Terms')  # type: ignore
    axes[1, 2].legend()  # type: ignore
    axes[1, 2].grid(True, alpha=0.3)  # type: ignore
    
    # Ra prediction tracking (new row)
    target_ra = dataset_params.get('Ra', 1e3) if dataset_params else 1e3
    if 'predicted_ra' in history and history['predicted_ra']:
        axes[2, 0].plot(history['predicted_ra'], label='Predicted Ra', linewidth=2, color='blue')  # type: ignore
        axes[2, 0].axhline(y=target_ra, color='red', linestyle='--', label=f'Target Ra ({target_ra:.1e})', linewidth=2)  # type: ignore
        axes[2, 0].set_title('Ra Prediction')  # type: ignore
        axes[2, 0].legend()  # type: ignore
        axes[2, 0].grid(True, alpha=0.3)  # type: ignore
        axes[2, 0].set_ylabel('Ra Value')  # type: ignore
    
    if 'ra_loss' in history and history['ra_loss']:
        axes[2, 1].semilogy([max(x, 1e-10) for x in history['ra_loss']], label='Ra Loss', linewidth=2, color='green')  # type: ignore
        axes[2, 1].set_title('Ra Loss (Log Scale)')  # type: ignore
        axes[2, 1].legend()  # type: ignore
        axes[2, 1].grid(True, alpha=0.3)  # type: ignore
        axes[2, 1].set_ylabel('MSE Loss')  # type: ignore
    
    # Ra prediction error
    if 'predicted_ra' in history and history['predicted_ra']:
        ra_errors = [abs(pred - target_ra) for pred in history['predicted_ra']]
        axes[2, 2].plot(ra_errors, label='|Predicted - Target|', linewidth=2, color='orange')  # type: ignore
        axes[2, 2].set_title('Ra Prediction Error')  # type: ignore
        axes[2, 2].legend()  # type: ignore
        axes[2, 2].grid(True, alpha=0.3)  # type: ignore
        axes[2, 2].set_ylabel('Absolute Error')  # type: ignore
    
    plt.suptitle(f'Complete Physics Analysis - Epoch {epoch}\nCurrent Predicted Ra: {avg_predicted_ra:.1e}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'complete_physics_results/epoch_{epoch:03d}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def get_time_step_indices(total_steps):
    """Get the three time step indices: 100, total/2, total-100"""
    
    # Ensure indices are within valid range
    start_idx = min(100, total_steps - 1)  # timestep 100
    mid_idx = total_steps // 2             # total/2
    end_idx = max(total_steps - 100, 0)    # total - 100
    
    return [start_idx, mid_idx, end_idx]

def create_epoch_comparison_figure(model, dataset, timestep_idx, epoch, device, save_dir):
    """Create a figure showing target, prediction, and error for all fields at a specific timestep"""
    
    # Field information
    field_names = ['U-Velocity', 'V-Velocity', 'Temperature', 'Pressure']
    field_codes = ['u', 'v', 't', 'p']
    
    # Create figure with subplots: 4 fields × 3 comparisons (target, prediction, error)
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(f'Epoch {epoch} - Time Step {timestep_idx}\nTarget vs Prediction vs Error', 
                 fontsize=16, fontweight='bold')
    
    try:
        # Get data sample
        input_state, target_state, _ = dataset[timestep_idx]
        input_state = input_state.unsqueeze(0).to(device)  # Add batch dimension
        target_state = target_state.to(device)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'predict_ra') and model.predict_ra:
                prediction, _ = model(input_state)  # Ignore Ra prediction for visualization
                prediction = prediction[:, :4]  # Take only the main 4 fields
            else:
                prediction = model(input_state)
            
            prediction = prediction.squeeze(0)  # Remove batch dimension
        
        # Convert tensors to numpy for plotting
        target_np = target_state.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        # Compute error
        error_np = pred_np - target_np
        
        for field_idx, (field_name, field_code) in enumerate(zip(field_names, field_codes)):
            
            # Get field data
            target_field = target_np[field_idx]
            pred_field = pred_np[field_idx] 
            error_field = error_np[field_idx]
            
            # Determine color limits for consistent scaling
            vmin_target = np.min(target_field)
            vmax_target = np.max(target_field)
            vmin_pred = np.min(pred_field)
            vmax_pred = np.max(pred_field)
            
            # Use consistent scale for target and prediction
            vmin_common = min(vmin_target, vmin_pred)
            vmax_common = max(vmax_target, vmax_pred)
            
            # Error scale (symmetric around zero)
            vmax_error = np.max(np.abs(error_field))
            vmin_error = -vmax_error
            
            # Target
            im1 = axes[field_idx, 0].imshow(target_field, cmap='seismic', 
                                           vmin=vmin_common, vmax=vmax_common, origin='lower')
            axes[field_idx, 0].set_title(f'{field_name} - Target', fontweight='bold')
            axes[field_idx, 0].set_xlabel('X')
            axes[field_idx, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[field_idx, 0], shrink=0.8)
            
            # Prediction
            im2 = axes[field_idx, 1].imshow(pred_field, cmap='seismic',
                                           vmin=vmin_common, vmax=vmax_common, origin='lower')
            axes[field_idx, 1].set_title(f'{field_name} - Prediction', fontweight='bold')
            axes[field_idx, 1].set_xlabel('X')
            axes[field_idx, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[field_idx, 1], shrink=0.8)
            
            # Error
            im3 = axes[field_idx, 2].imshow(error_field, cmap='seismic',
                                           vmin=vmin_error, vmax=vmax_error, origin='lower')
            axes[field_idx, 2].set_title(f'{field_name} - Error', fontweight='bold')
            axes[field_idx, 2].set_xlabel('X')
            axes[field_idx, 2].set_ylabel('Y')
            plt.colorbar(im3, ax=axes[field_idx, 2], shrink=0.8)
            
            # Add statistics as text
            mse = np.mean(error_field**2)
            max_error = np.max(np.abs(error_field))
            
            stats_text = f'MSE: {mse:.2e}\nMax |Error|: {max_error:.2e}'
            axes[field_idx, 2].text(0.02, 0.98, stats_text, transform=axes[field_idx, 2].transAxes,
                                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="white", alpha=0.8), fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'epoch_{epoch:03d}_timestep_{timestep_idx:03d}_comparison.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        plt.close()
        print(f"Error creating comparison figure for timestep {timestep_idx}: {e}")
        return None

def save_epoch_analysis(model, dataset, epoch, device, save_dir):
    """Save epoch analysis with target vs prediction vs error for multiple timesteps"""
    
    total_timesteps = len(dataset)
    timestep_indices = get_time_step_indices(total_timesteps)
    timestep_names = ['Early (t=100)', 'Middle (t=total/2)', 'Late (t=total-100)']
    
    print(f"  Creating epoch analysis for timesteps: {timestep_indices}")
    
    saved_files = []
    for i, timestep_idx in enumerate(timestep_indices):
        try:
            save_path = create_epoch_comparison_figure(
                model, dataset, timestep_idx, epoch, device, save_dir
            )
            if save_path:
                saved_files.append(save_path)
                print(f"    Saved {timestep_names[i]} (timestep {timestep_idx})")
        except Exception as e:
            print(f"    Error processing {timestep_names[i]} (timestep {timestep_idx}): {e}")
            continue
    
    return saved_files

if __name__ == "__main__":
    train_complete_physics_model() 