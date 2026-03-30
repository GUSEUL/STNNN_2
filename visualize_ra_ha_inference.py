import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from data import MatDataset
from models import MultiParamSurrogateModel
from train_and_infer_v4 import (
    preprocess_to_hdf5, CachedSequenceDataset
)

# =============================================================================
# 1. Multi-Parameter Inference (Ra & Ha) & Continuous Rollout
# =============================================================================
def predict_ra_ha_and_rollout(model, dataset, config, device, gt_p, rollout_len=100):
    model.eval()
    num_inference_samples = config.get('num_inference_samples', 20)
    num_restarts = config.get('num_restarts', 4)
    
    # 1. Optimization phase
    indices = np.linspace(0, len(dataset)-1, num_inference_samples, dtype=int)
    batch_input = torch.stack([dataset[i][0] for i in indices]).to(device)
    batch_target = torch.stack([dataset[i][1] for i in indices]).to(device)

    # Fix Q and Da to GT
    q_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Q'], device=device)
    d_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Da'], device=device)

    # Initialize Ra (Log-space) and Ha (Linear-space) as learnable parameters
    p_raw = torch.randn((num_restarts, 2), device=device, requires_grad=True) # [restarts, 2] -> [Ra_raw, Ha_raw]
    optimizer = optim.Adam([p_raw], lr=config['inference_lr'])
    
    ra_history = []
    ha_history = []
    
    print(f"  [Step 1] Optimizing Ra & Ha simultaneously (GT: Ra={gt_p['Ra']:.2e}, Ha={gt_p['Ha']:.2f})...")
    for step in range(config['inference_steps']):
        optimizer.zero_grad()
        
        # Mapping from sigmoid latent space to physical ranges
        latent = torch.sigmoid(p_raw)
        ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * latent[:, 0])
        ha = config['ha_min'] + (config['ha_max'] - config['ha_min']) * latent[:, 1]
        
        ra_e = ra.view(-1, 1).expand(-1, num_inference_samples).reshape(-1)
        ha_e = ha.view(-1, 1).expand(-1, num_inference_samples).reshape(-1)
        
        with autocast('cuda'):
            pred = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), ra_e, ha_e, q_fixed, d_fixed)
            l_data = (pred - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_inference_samples, -1).mean(dim=(1, 2))
            loss_total = (50.0 * l_data).mean()
        
        if torch.isnan(loss_total): break
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([p_raw], 0.1)
        optimizer.step()

        best_idx = torch.argmin(l_data).item()
        curr_ra = ra[best_idx].item()
        curr_ha = ha[best_idx].item()
        ra_history.append(curr_ra)
        ha_history.append(curr_ha)
        
        if step % 10 == 0:
            print(f"    [Step {step:4d}] Loss: {loss_total.item():.6f} | Pred Ra: {curr_ra:.2e} (GT:{gt_p['Ra']:.2e}) | Pred Ha: {curr_ha:.2f} (GT:{gt_p['Ha']:.2f})")

    # Final Parameters
    bi = torch.argmin(l_data).item()
    final_ra = ra[bi].detach()
    final_ha = ha[bi].detach()

    # 2. Continuous Rollout with Upsampling
    print(f"  [Step 2] Generating High-Res Rollout ({rollout_len} frames)...")
    target_res = 256
    
    with torch.no_grad(), autocast('cuda'):
        current_seq = dataset[0][0].unsqueeze(0).to(device)
        preds_hi = []
        gts_orig = []
        
        ra_param = final_ra.view(1)
        ha_param = final_ha.view(1)
        q_param = torch.tensor([gt_p['Q']], device=device)
        d_param = torch.tensor([gt_p['Da']], device=device)

        for i in range(rollout_len):
            out = model(current_seq, ra_param, ha_param, q_param, d_param)
            out_hi = F.interpolate(out, size=(target_res, target_res), mode='bicubic', align_corners=True)
            preds_hi.append(out_hi.squeeze(0).cpu().numpy())
            
            if i < len(dataset):
                gts_orig.append(dataset[i][1].cpu().numpy())
            
            current_seq = torch.cat([current_seq[:, 1:], out.unsqueeze(1)], dim=1)

    return final_ra.item(), final_ha.item(), ra_history, ha_history, np.array(gts_orig), np.array(preds_hi)

# =============================================================================
# 2. Dual Parameter Visualization
# =============================================================================
def save_dual_inference_visualizations(filename, gt_p, pred_p, ra_hist, ha_hist, gt_seq, pred_seq, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # 1. Convergence Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Ra Convergence
    ax1.plot(ra_hist, color='blue', lw=2, label='Predicted Ra')
    ax1.axhline(y=gt_p['Ra'], color='red', linestyle='--', label='Ground Truth Ra')
    ax1.set_yscale('log')
    ax1.set_title(f"Ra Convergence\nGT={gt_p['Ra']:.2e} | Pred={pred_p['Ra']:.2e}")
    ax1.set_xlabel("Steps"); ax1.set_ylabel("Ra"); ax1.legend()
    
    # Ha Convergence
    ax2.plot(ha_hist, color='green', lw=2, label='Predicted Ha')
    ax2.axhline(y=gt_p['Ha'], color='red', linestyle='--', label='Ground Truth Ha')
    ax2.set_title(f"Ha Convergence\nGT={gt_p['Ha']:.2f} | Pred={pred_p['Ha']:.2f}")
    ax2.set_xlabel("Steps"); ax2.set_ylabel("Ha"); ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_ra_ha_convergence.png"), dpi=150)
    plt.close()

    # 2. Animation
    num_frames = len(pred_seq)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    titles = ['u-velocity', 'v-velocity', 'Temperature', 'Pressure']
    
    im_objs = []
    for row in range(2):
        row_objs = []
        for col in range(4):
            res = 42 if row == 0 else 256
            im = axes[row, col].imshow(np.zeros((res, res)), cmap='jet', origin='lower')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(f"GT {titles[col]} (42x42)")
            else: axes[row, col].set_title(f"Pred {titles[col]} (High-Res 256x256)", fontweight='bold', color='blue')
            row_objs.append(im)
        im_objs.append(row_objs)

    def update(frame):
        for i in range(4):
            if frame < len(gt_seq):
                im_objs[0][i].set_data(gt_seq[frame, i])
                im_objs[0][i].set_clim(gt_seq[frame, i].min(), gt_seq[frame, i].max())
            im_objs[1][i].set_data(pred_seq[frame, i])
            im_objs[1][i].set_clim(pred_seq[frame, i].min(), pred_seq[frame, i].max())
            
        fig.suptitle(f"Simultaneous Ra-Ha Inference & Synthesis | Frame {frame:03d}\n"
                     f"Ra (GT: {gt_p['Ra']:.2e} | Pred: {pred_p['Ra']:.2e}) | "
                     f"Ha (GT: {gt_p['Ha']:.2f} | Pred: {pred_p['Ha']:.2f})", fontsize=16)
        return [item for sublist in im_objs for item in sublist]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
    anim.save(os.path.join(output_dir, f"{base_name}_dual_inference.gif"), writer=PillowWriter(fps=10))
    plt.close()
    print(f"  [Saved] Convergence and GIF for {base_name}")

# =============================================================================
# 3. Main
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoint_best_EG.pth' # User specified checkpoint
    output_dir = 'inference_results_ra_ha'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    model = MultiParamSurrogateModel(hidden=256, use_film=True).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"Model loaded from {checkpoint_path}")

    # Load test files
    base_path = os.path.join('data', 'EG')
    all_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    all_files = [f for f in sorted(all_files) if 'phi' not in f.lower()]
    random.seed(42); random.shuffle(all_files)
    test_files = all_files[:5] # Test on 5 samples

    inf_config = {
        'inference_steps': 1500, # Increased steps for dual-parameter
        'inference_lr': 0.005, 
        'ra_min': 100, 'ra_max': 1e8,
        'ha_min': 0, 'ha_max': 100,
        'num_restarts': 4,
        'num_inference_samples': 20
    }

    cache_dir = "cache_EG"
    for f in test_files:
        ds_m = MatDataset(f, device=device)
        ds_s = CachedSequenceDataset(preprocess_to_hdf5(f, cache_dir), device=device)
        
        ra_p, ha_p, ra_h, ha_h, gt_s, pred_s = predict_ra_ha_and_rollout(
            model, ds_s, inf_config, device, ds_m.params, rollout_len=50
        )
        
        pred_params = {'Ra': ra_p, 'Ha': ha_p}
        save_dual_inference_visualizations(f, ds_m.params, pred_params, ra_h, ha_h, gt_s, pred_s, output_dir)
        
        ra_err = abs(ra_p - ds_m.params['Ra']) / ds_m.params['Ra'] * 100
        ha_err = abs(ha_p - ds_m.params['Ha']) / (ds_m.params['Ha'] + 1e-8) * 100
        print(f"  [Result] {os.path.basename(f)} | Ra Err: {ra_err:.2f}% | Ha Err: {ha_err:.2f}%")

if __name__ == '__main__':
    main()
