"""
PhyCRNet Dynamic Multi-Parameter Solver V4.1 (Dynamic Frames & Recording)
=====================================================================
Modified version of v4.0 with frame selection recording logic.
Stores frame usage per timestep in JSON format for later visualization.
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
# 2. HDF5 Caching System
# =============================================================================
def get_file_hash(filepath):
    stat = os.stat(filepath)
    hash_input = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def preprocess_to_hdf5(mat_file_path, cache_dir, max_frames=5):
    if not HAS_H5PY: return None
    os.makedirs(cache_dir, exist_ok=True)
    file_hash = get_file_hash(mat_file_path)
    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    cache_path = os.path.join(cache_dir, f"{base_name}_{file_hash}_maxseq{max_frames}.h5")
    if os.path.exists(cache_path): return cache_path

    try:
        ds = MatDataset(mat_file_path, device='cpu')
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return None

    num_sequences = len(ds) - (max_frames - 1)
    if num_sequences <= 0: return None
    f0_sample, _, _ = ds[0]
    C, H, W = f0_sample.shape
    all_input_seqs = np.zeros((num_sequences, max_frames, C, H, W), dtype=np.float32)
    all_targets = np.zeros((num_sequences, C, H, W), dtype=np.float32)
    all_times = np.zeros((num_sequences, 1), dtype=np.float32)

    for i in range(num_sequences):
        seq_frames = []
        for s in range(max_frames):
            frame, _, _ = ds[i + s]
            seq_frames.append(frame.numpy())
        all_input_seqs[i] = np.stack(seq_frames)
        _, target, _ = ds[i + max_frames - 1]
        all_targets[i] = target.numpy()
        all_times[i] = (i + max_frames - 1) / ds.T
    
    with h5py.File(cache_path, 'w') as f:
        f.create_dataset('input_sequences', data=all_input_seqs, compression='lzf')
        f.create_dataset('targets', data=all_targets, compression='lzf')
        f.create_dataset('times', data=all_times, compression='lzf')
        p_grp = f.create_group('params')
        for k, v in ds.params.items():
            if isinstance(v, (int, float, np.number)): p_grp.attrs[k] = float(v)
        n_grp = f.create_group('nanofluid_props')
        for k, v in ds.nanofluid_props.items():
            if isinstance(v, (int, float, np.number)): n_grp.attrs[k] = float(v)
        norm_grp = f.create_group('norm_params')
        for k, (mu, std) in ds.norm_params.items():
            norm_grp.attrs[f'{k}_mu'] = float(mu)
            norm_grp.attrs[f'{k}_std'] = float(std)
    return cache_path

class CachedDynamicDataset(Dataset):
    def __init__(self, cache_path, device='cpu'):
        self.cache_path = cache_path
        self.device = device
        with h5py.File(cache_path, 'r') as f:
            self.length = f['input_sequences'].shape[0]
            _, _, _, self.ny, self.nx = f['input_sequences'].shape
            self.params = {k: float(f['params'].attrs[k]) for k in f['params'].attrs}
            self.nano_props = {k: float(f['nanofluid_props'].attrs[k]) for k in f['nanofluid_props'].attrs}
            self.norm_params = {}
            for k in ['u', 'v', 'p', 't']:
                self.norm_params[k] = (float(f['norm_params'].attrs[f'{k}_mu']), float(f['norm_params'].attrs[f'{k}_std']))
        self._file = None

    def __len__(self): return self.length
    def __getitem__(self, idx):
        if self._file is None: self._file = h5py.File(self.cache_path, 'r')
        inp = torch.from_numpy(self._file['input_sequences'][idx])
        tgt = torch.from_numpy(self._file['targets'][idx])
        t_val = torch.from_numpy(self._file['times'][idx])
        pd = {
            'Ra': torch.tensor(self.params.get('Ra', 1e4), dtype=torch.float32),
            'Ha': torch.tensor(self.params.get('Ha', 0.0), dtype=torch.float32),
            'Q':  torch.tensor(self.params.get('Q', 0.0), dtype=torch.float32),
            'Da': torch.tensor(self.params.get('Da', 1e-3), dtype=torch.float32)
        }
        return inp, tgt, t_val, pd

# =============================================================================
# 3. Physics Loss & Training
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

    def physics_residual_loss(self, inp_t, pred, r, h, q, d, steady=False):
        un_t, vn_t, tn_t, pn_t = torch.chunk(inp_t, 4, 1)
        un_x, vn_x, tn_x, pn_x = torch.chunk(pred, 4, 1)
        u_t, v_t, t_t, _ = self.unnorm(un_t, vn_t, tn_t, pn_t)
        u_x, v_x, t_x, p_x = self.unnorm(un_x, vn_x, tn_x, pn_x)
        ux_x, ux_y, ux_xx, ux_yy = self.compute_derivatives(u_x)
        vx_x, vx_y, vx_xx, vx_yy = self.compute_derivatives(v_x)
        tx_x, tx_y, tx_xx, tx_yy = self.compute_derivatives(t_x)
        px_x, px_y, _, _ = self.compute_derivatives(p_x)
        res_c = ux_x + vx_y
        dudt = 0 if steady else (u_x - u_t) / self.dt
        dvdt = 0 if steady else (v_x - v_t) / self.dt
        dtdt = 0 if steady else (t_x - t_t) / self.dt
        rb = d.view(-1, 1, 1, 1); rab = r.view(-1, 1, 1, 1)
        hab = h.view(-1, 1, 1, 1); qb = q.view(-1, 1, 1, 1)
        res_x = (dudt + u_x*ux_x + v_x*ux_y) - (-px_x + self.nu_r*self.Pr*(ux_xx+ux_yy) - (self.nu_r*self.Pr/rb)*u_x)
        res_y = (dvdt + u_x*vx_x + v_x*vx_y) - (-px_y + self.nu_r*self.Pr*(vx_xx+vx_yy) + self.beta_r*rab*self.Pr*t_x - (self.nu_r*self.Pr/rb)*v_x - (self.sigma_r*self.rho_r*hab**2*self.Pr)*v_x)
        res_e = (dtdt + u_x*tx_x + v_x*tx_y) - (self.alpha_r*(tx_xx+tx_yy) + self.cp_r*qb*t_x)
        return {'continuity': res_c.pow(2).mean(dim=[1, 2, 3]), 'momentum_x': res_x.pow(2).mean(dim=[1, 2, 3]),
                'momentum_y': res_y.pow(2).mean(dim=[1, 2, 3]), 'energy': res_e.pow(2).mean(dim=[1, 2, 3])}

def train_model(args, model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    sample_ds = train_loader.dataset.datasets[0] if hasattr(train_loader.dataset, 'datasets') else train_loader.dataset
    phys = MultiParamPhysicsLoss(sample_ds.params, sample_ds.nano_props, 
                                dt=sample_ds.params.get('dt', 0.0001), 
                                dx=1.0/(sample_ds.nx-1), dy=1.0/(sample_ds.ny-1)).to(device)
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (inp, tgt, t_val, pd) in enumerate(pbar):
            inp, tgt, t_val = inp.to(device), tgt.to(device), t_val.to(device)
            r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred = model(inp, t_val, r, h, q, d)
                loss_mse = F.mse_loss(pred, tgt)
            scaler.scale(loss_mse).backward(); scaler.step(optimizer); scaler.update()
            pbar.set_postfix({'mse': f"{loss_mse.item():.6f}"})
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for inp, tgt, t_val, pd in val_loader:
                inp, tgt, t_val = inp.to(device), tgt.to(device), t_val.to(device)
                r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
                pred = model(inp, t_val, r, h, q, d)
                v_loss += F.mse_loss(pred, tgt).item()
        v_loss /= len(val_loader); scheduler.step(v_loss)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), f'checkpoint_dynamic_best_{args.base_fluid}.pth')

# =============================================================================
# 4. Recording & Stats Logic
# =============================================================================
def record_frame_selection_stats(model, test_files, cache_dir, device, max_frames=5, output_prefix="EG"):
    """Records frame usage per timestep for visualization."""
    model.eval()
    os.makedirs("frame_stats", exist_ok=True)
    for i, f_path in enumerate(test_files[:3]):
        base_name = os.path.splitext(os.path.basename(f_path))[0]
        cp = preprocess_to_hdf5(f_path, cache_dir, max_frames=max_frames)
        ds = CachedDynamicDataset(cp, device=device); dl = DataLoader(ds, batch_size=1, shuffle=False)
        stats = []
        print(f"Recording frame stats for: {base_name}...")
        with torch.no_grad():
            for idx, (inp, tgt, t_val, pd) in enumerate(dl):
                inp, t_val = inp.to(device), t_val.to(device)
                r, h, q, d = pd['Ra'].to(device), pd['Ha'].to(device), pd['Q'].to(device), pd['Da'].to(device)
                _, p_count = model(inp, t_val, r, h, q, d, return_frame_count=True)
                stats.append({"step": idx, "t_norm": float(t_val.item()), "selected_frames": int(p_count.item())})
        save_path = f"frame_stats/{output_prefix}_{base_name}_frame_usage.json"
        with open(save_path, 'w') as f: json.dump(stats, f, indent=4)
        print(f"  [Stats Saved] {save_path}")

# =============================================================================
# 5. Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_fluid', default='EG'); parser.add_argument('--data_root', default='data')
    parser.add_argument('--epochs', type=int, default=100); parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_frames', type=int, default=5); parser.add_argument('--skip_train', action='store_true')
    args = parser.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_path = os.path.join(args.data_root, args.base_fluid)
    all_files = [f for f in sorted(glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)) if 'phi' not in f.lower()]
    random.seed(42); random.shuffle(all_files)
    n = len(all_files); train_files = all_files[:int(n*0.8)]; val_files = all_files[int(n*0.8):int(n*0.9)]; test_files = all_files[int(n*0.9):]
    
    cache_dir = f"cache_dynamic_{args.base_fluid}"
    def get_loader(files, shuffle=True):
        caches = [CachedDynamicDataset(preprocess_to_hdf5(f, cache_dir, args.max_frames)) for f in tqdm(files, desc="Caching") if preprocess_to_hdf5(f, cache_dir, args.max_frames)]
        return DataLoader(ConcatDataset(caches), batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    model = DynamicMultiParamSurrogateModel(max_frames=args.max_frames, hidden=256).to(device)
    if not args.skip_train:
        train_loader = get_loader(train_files); val_loader = get_loader(val_files, shuffle=False)
        train_model(args, model, train_loader, val_loader, device)
    
    model.load_state_dict(torch.load(f'checkpoint_dynamic_best_{args.base_fluid}.pth'))
    record_frame_selection_stats(model, test_files, cache_dir, device, args.max_frames, args.base_fluid)

if __name__ == '__main__':
    main()
