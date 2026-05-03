"""
PhyCRNet Dynamic Multi-Parameter Solver V4.2 (Epoch Logging)
=====================================================================
Added consistency-based Ra/Ha inference logging every 10 epochs.
Shows how well the model is learning the underlying physics during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
import numpy as np
import os
import hashlib
import argparse
import random
import glob
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from data import MatDataset, load_mat_file, extract_nanofluid_properties
from models import STNNN, MultiParamSurrogateModel

# =============================================================================
# 1. Dynamic Frame Selection Components
# =============================================================================

class FrameSelectorMLP(nn.Module):
    def __init__(self, input_ch=4, h=42, w=42, max_frames=5):
        super().__init__()
        self.max_frames = max_frames
        flat_dim = input_ch * h * w
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_frames)
        )
    
    def forward(self, x_last, t):
        x_flat = x_last.view(x_last.size(0), -1)
        inp = torch.cat([x_flat, t], dim=1)
        return self.mlp(inp)

class DynamicMultiParamSurrogateModel(nn.Module):
    def __init__(self, max_frames=5, hidden=256):
        super().__init__()
        self.max_frames = max_frames
        self.surrogate = MultiParamSurrogateModel(hidden=hidden)
        self.frame_selector = FrameSelectorMLP(max_frames=max_frames)

    def forward(self, x_seq, t, ra, ha, q, da, return_frame_count=False):
        B, S_max, C, H, W = x_seq.shape
        logits = self.frame_selector(x_seq[:, -1], t)
        probs = F.softmax(logits, dim=1)
        
        ra_n, ha_n, q_n, da_n = self.surrogate.normalize_params(ra, ha, q, da)
        param_vec = torch.stack([ra_n, ha_n, q_n, da_n], dim=-1)
        param_embed = self.surrogate.param_encoder(param_vec)
        
        x_reshaped = x_seq.view(B*S_max, C, H, W)
        z_reshaped = self.surrogate.stnnn.enc(x_reshaped)
        z_seq = z_reshaped.view(B, S_max, -1, H, W)
        
        out_lstm, _ = self.surrogate.stnnn.conv_lstm(z_seq)
        
        weighted_latent = 0
        for s in range(1, self.max_frames + 1):
            latent_s = self.surrogate.stnnn.residual_block(out_lstm[:, s-1])
            modulated_s = self.surrogate.film_layer(latent_s, param_embed)
            w = probs[:, s-1].view(B, 1, 1, 1)
            weighted_latent = weighted_latent + w * modulated_s
            
        output = self.surrogate.film_decoder(weighted_latent)
        
        if return_frame_count:
            predicted_count = torch.argmax(probs, dim=1) + 1
            return output, predicted_count
            
        return output

# =============================================================================
# 2. Physics & Consistency Loss
# =============================================================================
class MultiParamPhysicsLoss(nn.Module):
    def __init__(self, params, nanofluid_props=None, dt=0.0001, dx=1.0, dy=1.0):
        super().__init__()
        self.Pr = params.get('Pr', 0.71)
        self.dt, self.dx, self.dy = dt, dx, dy
        r = nanofluid_props if nanofluid_props else {}
        self.nu_r = r.get('nu_thnf_ratio', 1.0); self.sigma_r = r.get('sigma_thnf_ratio', 1.0)
        self.rho_r = r.get('rho_f_thnf_ratio', 1.0); self.beta_r = r.get('beta_thnf_ratio', 1.0)
        self.alpha_r = r.get('alpha_thnf_ratio', 1.0); self.cp_r = r.get('rhocp_f_thnf_ratio', 1.0)
        n = params.get('norm_params', {})
        self.u_mu, self.u_std = n.get('u', (0.0, 1.0)); self.v_mu, self.v_std = n.get('v', (0.0, 1.0))
        self.t_mu, self.t_std = n.get('t', (0.0, 1.0)); self.p_mu, self.p_std = n.get('p', (0.0, 1.0))

    def unnorm(self, un, vn, tn, pn):
        return (un * self.u_std + self.u_mu, vn * self.v_std + self.v_mu, 
                tn * self.t_std + self.t_mu, pn * self.p_std + self.p_mu)

    def compute_derivatives(self, f):
        fx = torch.gradient(f, dim=-1)[0] / self.dx; fy = torch.gradient(f, dim=-2)[0] / self.dy
        fxx = torch.gradient(fx, dim=-1)[0] / self.dx; fyy = torch.gradient(fy, dim=-2)[0] / self.dy
        return fx, fy, fxx, fyy

    def physics_residual_loss(self, inp_t, pred, r, h, q, d):
        un_t, vn_t, tn_t, pn_t = torch.chunk(inp_t, 4, 1)
        un_x, vn_x, tn_x, pn_x = torch.chunk(pred, 4, 1)
        u_t, v_t, t_t, _ = self.unnorm(un_t, vn_t, tn_t, pn_t)
        u_x, v_x, t_x, p_x = self.unnorm(un_x, vn_x, tn_x, pn_x)
        ux_x, ux_y, ux_xx, ux_yy = self.compute_derivatives(u_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(v_x)
        tx_x, tx_y, tx_xx, tx_yy = self.compute_derivatives(t_x)
        px_x, px_y, _, _ = self.compute_derivatives(p_x)
        res_c = ux_x + vx_y
        dudt = (u_x - u_t) / self.dt; dvdt = (v_x - v_t) / self.dt; dtdt = (t_x - t_t) / self.dt
        rb = d.view(-1, 1, 1, 1); rab = r.view(-1, 1, 1, 1)
        hab = h.view(-1, 1, 1, 1); qb = q.view(-1, 1, 1, 1)
        res_x = (dudt + u_x*ux_x + v_x*ux_y) - (-px_x + self.nu_r*self.Pr*(ux_xx+ux_yy) - (self.nu_r*self.Pr/rb)*u_x)
        res_y = (dvdt + u_x*vx_x + v_x*vx_y) - (-px_y + self.nu_r*self.Pr*(vx_xx+vx_yy) + self.beta_r*rab*self.Pr*t_x - (self.nu_r*self.Pr/rb)*v_x - (self.sigma_r*self.rho_r*hab**2*self.Pr)*v_x)
        res_e = (dtdt + u_x*tx_x + v_x*tx_y) - (self.alpha_r*(tx_xx+tx_yy) + self.cp_r*qb*t_x)
        return {'c': res_c, 'x': res_x, 'y': res_y, 'e': res_e}

    def ra_consistency_loss(self, un, vn, pn_x, un_x, vn_x, tn_x, d_val, h_val):
        z_t = torch.zeros_like(un)
        u, v, _, _ = self.unnorm(un, vn, z_t, z_t)
        ux, vx, tx, px = self.unnorm(un_x, vn_x, tn_x, pn_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(vx)
        px_y, _, _, _ = self.compute_derivatives(px)
        dvdt = (vx - v) / self.dt
        rb = d_val.view(-1, 1, 1, 1); hab = h_val.view(-1, 1, 1, 1)
        rhs = dvdt + ux*vx_x + vx*vx_y + px_y - self.nu_r*self.Pr*(vx_xx+vx_yy) + (self.nu_r*self.Pr/rb)*vx + (self.sigma_r*self.rho_r*hab**2*self.Pr)*vx
        ra_inferred = rhs / (self.beta_r * self.Pr * tx + 1e-7)
        return torch.clamp(ra_inferred, 10.0, 1e9).mean(dim=[1, 2, 3])

    def ha_consistency_loss(self, un, vn, pn_x, un_x, vn_x, tn_x, r_val, d_val):
        z_t = torch.zeros_like(un)
        u, v, _, _ = self.unnorm(un, vn, z_t, z_t)
        ux, vx, tx, px = self.unnorm(un_x, vn_x, tn_x, pn_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(vx)
        px_y, _, _, _ = self.compute_derivatives(px)
        dvdt = (vx - v) / self.dt
        rb = d_val.view(-1, 1, 1, 1); rab = r_val.view(-1, 1, 1, 1)
        rhs = -(dvdt + ux*vx_x + vx*vx_y + px_y - self.nu_r*self.Pr*(vx_xx+vx_yy) - self.beta_r*rab*self.Pr*tx + (self.nu_r*self.Pr/rb)*vx)
        ha_sq = rhs / (self.sigma_r * self.rho_r * self.Pr * vx + 1e-7)
        return torch.sqrt(torch.clamp(ha_sq, 0, 1e5)).mean(dim=[1, 2, 3])

# =============================================================================
# 3. HDF5 Caching & Dataset
# =============================================================================
def get_file_hash(filepath):
    stat = os.stat(filepath); hash_input = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def preprocess_to_hdf5(mat_file_path, cache_dir, max_frames=5):
    if not HAS_H5PY: return None
    os.makedirs(cache_dir, exist_ok=True); file_hash = get_file_hash(mat_file_path)
    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    cache_path = os.path.join(cache_dir, f"{base_name}_{file_hash}_maxseq{max_frames}.h5")
    if os.path.exists(cache_path): return cache_path
    try:
        ds = MatDataset(mat_file_path, device='cpu')
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}"); return None
    num_seq = len(ds) - (max_frames - 1)
    if num_seq <= 0: return None
    f0, _, _ = ds[0]; C, H, W = f0.shape
    all_inp = np.zeros((num_seq, max_frames, C, H, W), dtype=np.float32)
    all_tgt = np.zeros((num_seq, C, H, W), dtype=np.float32); all_t = np.zeros((num_seq, 1), dtype=np.float32)
    for i in range(num_seq):
        seq = [ds[i+s][0].numpy() for s in range(max_frames)]
        all_inp[i] = np.stack(seq); all_tgt[i] = ds[i+max_frames-1][1].numpy(); all_t[i] = (i+max_frames-1)/ds.T
    with h5py.File(cache_path, 'w') as f:
        f.create_dataset('input_sequences', data=all_inp, compression='lzf')
        f.create_dataset('targets', data=all_tgt, compression='lzf')
        f.create_dataset('times', data=all_t, compression='lzf')
        p_grp = f.create_group('params'); n_grp = f.create_group('nanofluid_props'); norm_grp = f.create_group('norm_params')
        for k, v in ds.params.items():
            if isinstance(v, (int, float, np.number)): p_grp.attrs[k] = float(v)
        for k, v in ds.nanofluid_props.items():
            if isinstance(v, (int, float, np.number)): n_grp.attrs[k] = float(v)
        for k, (mu, std) in ds.norm_params.items():
            norm_grp.attrs[f'{k}_mu'] = float(mu); norm_grp.attrs[f'{k}_std'] = float(std)
    return cache_path

class CachedDynamicDataset(Dataset):
    def __init__(self, cache_path, device='cpu'):
        self.cache_path = cache_path; self.device = device
        with h5py.File(cache_path, 'r') as f:
            self.length = f['input_sequences'].shape[0]; _, _, _, self.ny, self.nx = f['input_sequences'].shape
            self.params = {k: float(f['params'].attrs[k]) for k in f['params'].attrs}
            self.nano_props = {k: float(f['nanofluid_props'].attrs[k]) for k in f['nanofluid_props'].attrs}
            self.norm_params = {k: (float(f['norm_params'].attrs[f'{k}_mu']), float(f['norm_params'].attrs[f'{k}_std'])) for k in ['u', 'v', 'p', 't']}
        self._file = None
    def __len__(self): return self.length
    def __getitem__(self, idx):
        if self._file is None: self._file = h5py.File(self.cache_path, 'r')
        inp = torch.from_numpy(self._file['input_sequences'][idx]); tgt = torch.from_numpy(self._file['targets'][idx])
        t_val = torch.from_numpy(self._file['times'][idx])
        pd = {k: torch.tensor(self.params.get(k, 0.0), dtype=torch.float32) for k in ['Ra', 'Ha', 'Q', 'Da']}
        return inp, tgt, t_val, pd

# =============================================================================
# 4. Training Loop
# =============================================================================
def train_model(args, model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    
    sample_ds = val_loader.dataset.datasets[0] if hasattr(val_loader.dataset, 'datasets') else val_loader.dataset
    phys = MultiParamPhysicsLoss(sample_ds.params, sample_ds.nano_props, dt=sample_ds.params.get('dt', 1e-4), 
                                dx=1.0/(sample_ds.nx-1), dy=1.0/(sample_ds.ny-1)).to(device)
    
    best_v_loss = float('inf')
    for epoch in range(args.epochs):
        model.train(); t_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inp, tgt, t_val, pd in pbar:
            inp, tgt, t_val = inp.to(device), tgt.to(device), t_val.to(device)
            r, h, q, d = [pd[k].to(device) for k in ['Ra', 'Ha', 'Q', 'Da']]
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred = model(inp, t_val, r, h, q, d)
                loss = F.mse_loss(pred, tgt)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            t_loss += loss.item(); pbar.set_postfix({'mse': f"{loss.item():.6f}"})

        # Validation & Logging
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for i, (inp, tgt, t_val, pd) in enumerate(val_loader):
                inp, tgt, t_val = inp.to(device), tgt.to(device), t_val.to(device)
                r, h, q, d = [pd[k].to(device) for k in ['Ra', 'Ha', 'Q', 'Da']]
                pred = model(inp, t_val, r, h, q, d)
                v_loss += F.mse_loss(pred, tgt).item()
                
                # Every 10 epochs, print consistency results for the first batch of validation
                if i == 0 and (epoch + 1) % 10 == 0:
                    avg_train_loss = t_loss / len(train_loader)
                    avg_val_loss = v_loss # Will be divided by len(val_loader) later, using partial here
                    un, vn = torch.chunk(inp[:, -1], 4, 1)[:2]; unx, vnx, tnx, pnx = torch.chunk(pred, 4, 1)
                    inf_ra = phys.ra_consistency_loss(un, vn, pnx, unx, vnx, tnx, d, h)
                    inf_ha = phys.ha_consistency_loss(un, vn, pnx, unx, vnx, tnx, r, d)
                    
                    print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss Pending...")
                    print(f"  Ra | GT: {r[0].item():.2e} | Pred: {inf_ra[0].item():.2e} | Err: {abs(r[0]-inf_ra[0])/r[0]*100:.2f}%")
                    print(f"  Ha | GT: {h[0].item():.2f} | Pred: {inf_ha[0].item():.2f} | Err: {abs(h[0]-inf_ha[0])/(h[0]+1e-8)*100:.2f}%")
        
        v_loss /= len(val_loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Final Val Loss for Epoch {epoch+1}: {v_loss:.6f}\n")
        
        scheduler.step(v_loss)
        if v_loss < best_v_loss:
            best_v_loss = v_loss; torch.save(model.state_dict(), f'checkpoint_dynamic_best_{args.base_fluid}.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_fluid', default='EG')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_files = [f for f in sorted(glob.glob(os.path.join('data', args.base_fluid, "**", "*.mat"), recursive=True)) if 'phi' not in f.lower()]
    random.seed(42); random.shuffle(all_files); n = len(all_files)
    train_f = all_files[:int(n*0.8)]; val_f = all_files[int(n*0.8):int(n*0.9)]
    
    cache_dir = f"cache_dynamic_{args.base_fluid}"
    def get_loader(files):
        ds_list = [CachedDynamicDataset(preprocess_to_hdf5(f, cache_dir)) for f in files if preprocess_to_hdf5(f, cache_dir)]
        return DataLoader(ConcatDataset(ds_list), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = DynamicMultiParamSurrogateModel(max_frames=5).to(device)
    
    # Resume logic
    checkpoint_to_load = args.resume
    if not checkpoint_to_load:
        # Check for default checkpoint if no specific one is provided
        default_ckpt = f'checkpoint_dynamic_best_{args.base_fluid}.pth'
        if os.path.exists(default_ckpt):
            checkpoint_to_load = default_ckpt
            
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"Loading checkpoint from {checkpoint_to_load}...")
        model.load_state_dict(torch.load(checkpoint_to_load, weights_only=True))

    train_model(args, model, get_loader(train_f), get_loader(val_f), device)

if __name__ == '__main__':
    main()
