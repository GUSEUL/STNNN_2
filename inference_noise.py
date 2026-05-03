import torch
import torch.nn as nn
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
import sys
import json
import csv
import scipy.io as sio

from data import MatDataset
from models import MultiParamSurrogateModel
from train_and_infer_v4 import (
    preprocess_to_hdf5, CachedSequenceDataset
)

# =============================================================================
# 1. Data Saving Utilities
# =============================================================================
def save_inference_history(base_name, ra_h, ha_h, loss_h, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    steps = np.arange(len(ra_h))
    
    csv_path = os.path.join(output_dir, f"{base_name}_inference_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Ra_Predicted', 'Ha_Predicted', 'MSE_Loss'])
        for i in range(len(ra_h)):
            writer.writerow([i, ra_h[i], ha_h[i], loss_h[i]])

    json_path = os.path.join(output_dir, f"{base_name}_inference_history.json")
    data_dict = {"case": base_name, "history": {"step": steps.tolist(), "ra_pred": ra_h, "ha_pred": ha_h, "loss": loss_h}}
    with open(json_path, 'w') as f: json.dump(data_dict, f, indent=4)

    mat_path = os.path.join(output_dir, f"{base_name}_inference_history.mat")
    sio.savemat(mat_path, {'step': steps, 'ra_pred': np.array(ra_h), 'ha_pred': np.array(ha_h), 'loss': np.array(loss_h)})

# =============================================================================
# 2. Parameter Inference (Ra & Ha) & Rollout (Normalized Space)
# =============================================================================
def predict_ra_ha_and_rollout(model, dataset, config, device, gt_p, rollout_len=100):
    model.eval()
    num_inference_samples = config.get('num_inference_samples', 20)
    num_restarts = config.get('num_restarts', 4)
    noise_level = config.get('noise_level', 0.0)

    # 1. Optimization phase
    indices = np.linspace(0, len(dataset)-1, num_inference_samples, dtype=int)
    batch_input = torch.stack([dataset[i][0] for i in indices]).to(device)
    batch_target = torch.stack([dataset[i][1] for i in indices]).to(device)

    if noise_level > 0:
        print(f"  [Robustness] Adding {noise_level*100:.1f}% Gaussian noise to optimization targets...")
        noise = torch.randn_like(batch_target) * noise_level
        batch_target = batch_target + noise

    q_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Q'], device=device)
    d_fixed = torch.full((num_restarts * num_inference_samples,), gt_p['Da'], device=device)

    p_raw = torch.randn((num_restarts, 2), device=device, requires_grad=True)
    optimizer = optim.Adam([p_raw], lr=config['inference_lr'])

    ra_history, ha_history, loss_history = [], [], []

    print(f"  [Step 1] Optimizing Ra & Ha for Standard Model (2500 steps)...")
    pbar_opt = tqdm(range(config['inference_steps']), desc="Optimization", mininterval=0.5)
    for step in pbar_opt:
        optimizer.zero_grad()
        latent = torch.sigmoid(p_raw)
        ra = 10**(np.log10(config['ra_min']) + (np.log10(config['ra_max']) - np.log10(config['ra_min'])) * latent[:, 0])
        ha = config['ha_min'] + (config['ha_max'] - config['ha_min']) * latent[:, 1]
        ra_e = ra.view(-1, 1).expand(-1, num_inference_samples).reshape(-1)
        ha_e = ha.view(-1, 1).expand(-1, num_inference_samples).reshape(-1)

        with autocast(device_type='cuda'):
            pred = model(batch_input.repeat(num_restarts, 1, 1, 1, 1), ra_e, ha_e, q_fixed, d_fixed)
            l_data = (pred - batch_target.repeat(num_restarts, 1, 1, 1)).pow(2).view(num_restarts, num_inference_samples, -1).mean(dim=(1, 2))
            loss_total = (50.0 * l_data).mean()

        loss_total.backward()
        optimizer.step()

        best_idx = torch.argmin(l_data).item()
        ra_history.append(ra[best_idx].item())
        ha_history.append(ha[best_idx].item())
        loss_history.append(loss_total.item())

        if step % 50 == 0:
            pbar_opt.set_postfix({
                "Loss": f"{loss_total.item():.4f}",
                "Ra(P/G)": f"{ra[best_idx].item():.1e}/{gt_p['Ra']:.1e}",
                "Ha(P/G)": f"{ha[best_idx].item():.1f}/{gt_p['Ha']:.1f}"
            })

        if step % 500 == 0:
            # Explicit print for log persistence
            tqdm.write(f"    Step {step:4d} | Loss: {loss_total.item():.6f} | Ra(P/G): {ra[best_idx].item():.1e}/{gt_p['Ra']:.1e} | Ha(P/G): {ha[best_idx].item():.1f}/{gt_p['Ha']:.1f}")
    final_ra, final_ha = ra[best_idx].detach(), ha[best_idx].detach()

    # 2. Continuous Rollout (Normalized Space as per v4 standard)
    print(f"  [Step 2] Generating Rollout ({rollout_len} frames)...")
    with torch.no_grad(), autocast(device_type='cuda'):
        current_seq = dataset[0][0].unsqueeze(0).to(device)
        preds_orig, gts_orig = [], []
        ra_param, ha_param = final_ra.view(1), final_ha.view(1)
        q_param, d_param = torch.tensor([gt_p['Q']], device=device), torch.tensor([gt_p['Da']], device=device)

        for i in tqdm(range(rollout_len), desc="Rollout"):
            out = model(current_seq, ra_param, ha_param, q_param, d_param)
            preds_orig.append(out.squeeze(0).cpu().numpy())
            if i < len(dataset):
                gts_orig.append(dataset[i][1].cpu().numpy())
            else:
                gts_orig.append(np.full((4, 42, 42), np.nan))
            current_seq = torch.cat([current_seq[:, 1:], out.unsqueeze(1)], dim=1)

    return final_ra.item(), final_ha.item(), ra_history, ha_history, loss_history, np.array(gts_orig), np.array(preds_orig)

# =============================================================================
# 3. Visualization (Simplified 2x4 Layout)
# =============================================================================
def save_comprehensive_visualizations(filename, gt_p, pred_p, ra_hist, ha_hist, gt_seq, pred_orig, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    num_frames = len(pred_orig)

    # 1. Metrics Plot
    fig_m, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.plot(ra_hist, 'b-', label=f"Pred Ra: {pred_p['Ra']:.2e}")
    ax1.axhline(y=gt_p['Ra'], color='r', linestyle='--', label=f"GT Ra: {gt_p['Ra']:.2e}")
    ax1.set_yscale('log'); ax1.set_title(f"Ra Convergence"); ax1.legend()
    ax2.plot(ha_hist, 'g-', label=f"Pred Ha: {pred_p['Ha']:.2f}")
    ax2.axhline(y=gt_p['Ha'], color='r', linestyle='--', label=f"GT Ha: {gt_p['Ha']:.2f}")
    ax2.set_title(f"Ha Convergence"); ax2.legend()
    valid_mask = ~np.isnan(gt_seq[:, 0, 0, 0])
    errors = [np.mean((pred_orig[i] - gt_seq[i])**2) for i in range(num_frames) if valid_mask[i]]
    ax3.plot(np.where(valid_mask)[0], errors, 'k-'); ax3.set_yscale('log'); ax3.set_title("Normalized MSE over Rollout")
    plt.savefig(os.path.join(output_dir, f"{base_name}_metrics.png"), dpi=150); plt.close(fig_m)

    # 2. Animation (Stride 10)
    stride = 10
    indices = np.arange(0, num_frames, stride)
    print(f"  [Step 3] Rendering Animation ({len(indices)} frames)...")
    fig_anim, axes_anim = plt.subplots(2, 4, figsize=(20, 10))
    titles = ['u-velocity', 'v-velocity', 'Temperature (T)', 'Pressure (p)']

    # Pre-create imshow objects for performance
    im_objs = []
    for row in range(2):
        row_objs = []
        for col in range(4):
            res = 42 if row == 0 else 256
            im = axes_anim[row, col].imshow(np.zeros((res, res)), cmap='jet', origin='lower')
            axes_anim[row, col].set_xticks([]); axes_anim[row, col].set_yticks([])
            axes_anim[row, col].set_title(f"{'GT' if row==0 else 'Pred'} {titles[col]}")
            row_objs.append(im)
        im_objs.append(row_objs)

    def update(idx_idx):
        frame = indices[idx_idx]
        # GT (Row 0)
        if not np.isnan(gt_seq[frame, 0, 0, 0]):
            for i in range(4):
                im_objs[0][i].set_data(gt_seq[frame, i])
                im_objs[0][i].set_clim(gt_seq[frame, i].min(), gt_seq[frame, i].max())
        
        # Pred (Row 1) - Upsample for visualization quality
        out_low = torch.from_numpy(pred_orig[frame]).unsqueeze(0)
        out_hi = F.interpolate(out_low, size=(256, 256), mode='bicubic', align_corners=True).squeeze(0).numpy()
        for i in range(4):
            im_objs[1][i].set_data(out_hi[i])
            im_objs[1][i].set_clim(out_hi[i].min(), out_hi[i].max())
        
        fig_anim.suptitle(f"Frame {frame:03d} | Ra(GT:{gt_p['Ra']:.2e} Pred:{pred_p['Ra']:.2e}) Ha(GT:{gt_p['Ha']:.2f} Pred:{pred_p['Ha']:.2f})", fontsize=16)
        return [item for sublist in im_objs for item in sublist]

    anim = FuncAnimation(fig_anim, update, frames=len(indices), interval=100, blit=True)
    anim.save(os.path.join(output_dir, f"{base_name}_rollout.gif"), writer=PillowWriter(fps=10))
    plt.close(fig_anim)

# =============================================================================
# 4. Main
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoint_best_EG.pth'
    noise_val = 0.01; out_dir = 'robustness_test_visuals_added_noise'; mat = 'EG'
    if len(sys.argv) > 1: noise_val = float(sys.argv[1])
    if len(sys.argv) > 2: out_dir = sys.argv[2]
    if len(sys.argv) > 3: mat = sys.argv[3]

    model = MultiParamSurrogateModel(hidden=256, use_film=True).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    else:
        print(f"Warning: {checkpoint_path} not found.")

    base_path = os.path.join('data', mat)
    all_files = [f for f in sorted(glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)) if 'phi' not in f.lower()]
    ra_f = [f for f in all_files if '_Ra_' in f]; ha_f = [f for f in all_files if '_Ha_' in f]; oth = [f for f in all_files if '_Ra_' not in f and '_Ha_' not in f]
    random.seed(42); 
    
    # Safe sampling
    test_files = []
    if len(ra_f) >= 5: test_files += random.sample(ra_f, 5)
    else: test_files += ra_f
    if len(ha_f) >= 5: test_files += random.sample(ha_f, 5)
    else: test_files += ha_f
    if len(oth) >= 2: test_files += random.sample(oth, 2)
    else: test_files += oth

    inf_config = {'inference_steps': 2500, 'inference_lr': 0.005, 'ra_min': 100, 'ra_max': 1e8, 'ha_min': 0, 'ha_max': 100, 'num_restarts': 4, 'num_inference_samples': 20, 'noise_level': noise_val}

    results = []
    history_dir = os.path.join(out_dir, "inference_history")
    os.makedirs(history_dir, exist_ok=True)

    for f in test_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        ds_m = MatDataset(f, device=device)
        ds_s = CachedSequenceDataset(preprocess_to_hdf5(f, "cache_EG"), device=device)
        ra_p, ha_p, ra_h, ha_h, loss_h, gt_s, pred_o = predict_ra_ha_and_rollout(model, ds_s, inf_config, device, ds_m.params, rollout_len=len(ds_s))
        save_inference_history(base_name, ra_h, ha_h, loss_h, history_dir)
        save_comprehensive_visualizations(f, ds_m.params, {'Ra': ra_p, 'Ha': ha_p}, ra_h, ha_h, gt_s, pred_o, out_dir)
        err_ra = abs(ra_p-ds_m.params['Ra'])/ds_m.params['Ra']*100
        err_ha = abs(ha_p-ds_m.params['Ha'])/(ds_m.params['Ha']+1e-8)*100
        results.append({'name': os.path.basename(f), 'ra_gt': ds_m.params['Ra'], 'ra_pred': ra_p, 'ha_gt': ds_m.params['Ha'], 'ha_pred': ha_p, 'ra_err': err_ra, 'ha_err': err_ha})
        print(f"  [Done] {os.path.basename(f)} | Ra GT:{results[-1]['ra_gt']:.2e} Pred:{results[-1]['ra_pred']:.2e} | Ha GT:{results[-1]['ha_gt']:.2f} Pred:{results[-1]['ha_pred']:.2f}")

    with open(os.path.join(out_dir, "inference_summary.log"), "w") as log:
        header = f"{'File':<25} | {'GT Ra':<10} | {'Pred Ra':<10} | {'Ra Err%':<8} | {'GT Ha':<8} | {'Pred Ha':<8} | {'Ha Err%':<8}\n"
        log.write(header + "-"*100 + "\n")
        for r in results:
            log.write(f"{r['name']:<25} | {r['ra_gt']:<10.2e} | {r['ra_pred']:<10.2e} | {r['ra_err']:<8.2f} | {r['ha_gt']:<8.2f} | {r['ha_pred']:<8.2f} | {r['ha_err']:<8.2f}\n")
        if results:
            log.write("-"*100 + f"\nAVERAGE ERROR | Ra: {np.mean([r['ra_err'] for r in results]):.2f}% | Ha: {np.mean([r['ha_err'] for r in results]):.2f}%\n")

if __name__ == '__main__':
    main()
