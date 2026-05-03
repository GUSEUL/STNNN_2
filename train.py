import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
import numpy as np
import os
import argparse
import random
import glob
import json
from tqdm import tqdm

from data import MatDataset
from train_and_infer_v4 import preprocess_to_hdf5, CachedSequenceDataset

# 가속 설정
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# =============================================================================
# 1. 고속 물리 손실 (구조는 동일, 연산만 가속)
# =============================================================================
class FastPhysicsLoss(nn.Module):
    def __init__(self, params, nanofluid_props=None, dt=0.0001, dx=1.0, dy=1.0):
        super().__init__()
        self.Pr = params.get('Pr', 0.71)
        self.dt, self.dx, self.dy = dt, dx, dy
        
        r = nanofluid_props if nanofluid_props else {}
        self.nu_r = r.get('nu_thnf_ratio', 1.0)
        self.sigma_r = r.get('sigma_thnf_ratio', 1.0)
        self.rho_r = r.get('rho_f_thnf_ratio', 1.0)
        self.beta_r = r.get('beta_thnf_ratio', 1.0)
        self.alpha_r = r.get('alpha_thnf_ratio', 1.0)
        self.cp_r = r.get('rhocp_f_thnf_ratio', 1.0)
        
        n = params.get('norm_params', {})
        self.u_mu, self.u_std = n.get('u', (0.0, 1.0))
        self.v_mu, self.v_std = n.get('v', (0.0, 1.0))
        self.t_mu, self.t_std = n.get('t', (0.0, 1.0))
        self.p_mu, self.p_std = n.get('p', (0.0, 1.0))

        self.register_buffer('kx', torch.tensor([[[-0.5, 0, 0.5]]], dtype=torch.float32).view(1, 1, 1, 3) / dx)
        self.register_buffer('ky', torch.tensor([[[-0.5], [0], [0.5]]], dtype=torch.float32).view(1, 1, 3, 1) / dy)
        self.register_buffer('kxx', torch.tensor([[[1, -2, 1]]], dtype=torch.float32).view(1, 1, 1, 3) / (dx**2))
        self.register_buffer('kyy', torch.tensor([[[1], [-2], [1]]], dtype=torch.float32).view(1, 1, 3, 1) / (dy**2))

    def diff_x(self, f): return F.conv2d(F.pad(f, (1, 1, 0, 0), mode='replicate'), self.kx)
    def diff_y(self, f): return F.conv2d(F.pad(f, (0, 0, 1, 1), mode='replicate'), self.ky)
    def diff_xx(self, f): return F.conv2d(F.pad(f, (1, 1, 0, 0), mode='replicate'), self.kxx)
    def diff_yy(self, f): return F.conv2d(F.pad(f, (0, 0, 1, 1), mode='replicate'), self.kyy)

    def unnorm(self, un, vn, tn, pn):
        return (un * self.u_std + self.u_mu, vn * self.v_std + self.v_mu, 
                tn * self.t_std + self.t_mu, pn * self.p_std + self.p_mu)

    def physics_residual_loss(self, inp_t, pred, r, h, q, d, steady=False):
        un_t, vn_t, tn_t, pn_t = torch.chunk(inp_t, 4, 1)
        un_x, vn_x, tn_x, pn_x = torch.chunk(pred, 4, 1)
        u_t, v_t, t_t, _ = self.unnorm(un_t, vn_t, tn_t, pn_t)
        u_x, v_x, t_x, p_x = self.unnorm(un_x, vn_x, tn_x, pn_x)

        ux_x = self.diff_x(u_x); ux_y = self.diff_y(u_x)
        ux_xx = self.diff_xx(u_x); ux_yy = self.diff_yy(u_x)
        vx_x = self.diff_x(v_x); vx_y = self.diff_y(v_x)
        vx_xx = self.diff_xx(v_x); vx_yy = self.diff_yy(v_x)
        tx_x = self.diff_x(t_x); tx_y = self.diff_y(t_x)
        tx_xx = self.diff_xx(t_x); tx_yy = self.diff_yy(t_x)
        px_x = self.diff_x(p_x); px_y = self.diff_y(p_x)

        res_c = ux_x + vx_y
        dudt = (u_x - u_t) / self.dt if not steady else 0
        dvdt = (v_x - v_t) / self.dt if not steady else 0
        dtdt = (t_x - t_t) / self.dt if not steady else 0

        rb = torch.clamp(d.view(-1, 1, 1, 1), min=1e-6)
        rab = r.view(-1, 1, 1, 1); hab = h.view(-1, 1, 1, 1); qb = q.view(-1, 1, 1, 1)

        res_x = (dudt + u_x*ux_x + v_x*ux_y) - (-px_x + self.nu_r*self.Pr*(ux_xx+ux_yy) - (self.nu_r*self.Pr/rb)*u_x)
        res_y = (dvdt + u_x*vx_x + v_x*vx_y) - (-px_y + self.nu_r*self.Pr*(vx_xx+vx_yy) + self.beta_r*rab*self.Pr*t_x - (self.nu_r*self.Pr/rb)*v_x - (self.sigma_r*self.rho_r*hab**2*self.Pr)*v_x)
        res_e = (dtdt + u_x*tx_x + v_x*tx_y) - (self.alpha_r*(tx_xx+tx_yy) + self.cp_r*qb*t_x)

        return {
            'continuity': res_c.pow(2).mean(dim=[1, 2, 3]),
            'momentum_x': res_x.pow(2).mean(dim=[1, 2, 3]),
            'momentum_y': res_y.pow(2).mean(dim=[1, 2, 3]),
            'energy':     res_e.pow(2).mean(dim=[1, 2, 3])
        }

# =============================================================================
# 2. 물리 손실 정규화 가중치 계산 (v4 방식)
# =============================================================================
def calculate_physics_normalization(model, dataloader, phys_fn, device, num_batches=3):
    print(f"\n  [Setup] Calculating Physics Normalization Weights...")
    model.eval()
    accum = {'continuity': [], 'momentum_x': [], 'momentum_y': [], 'energy': []}
    with torch.no_grad():
        for i, (inp, tgt, pd) in enumerate(dataloader):
            if i >= num_batches: break
            inp = inp.to(device)
            r, h, q, d = [pd[k].to(device) for k in ['Ra', 'Ha', 'Q', 'Da']]
            with autocast('cuda'):
                pred = model(inp, r, h, q, d)
            p_losses = phys_fn.physics_residual_loss(inp[:, -1].float(), pred.float(), r, h, q, d)
            for k in accum.keys():
                val = p_losses[k].mean().item()
                if np.isfinite(val) and val > 0: accum[k].append(val)
    weights = {}
    for k, vals in accum.items():
        mean_val = np.mean(vals) if vals else 1.0
        weights[k] = 1.0 / (mean_val + 1e-9)
        print(f"    - {k:10s} | Norm Weight: {weights[k]:.2e}")
    return weights

# =============================================================================
# 3. 경량화 모델 (LightSurrogateModel)
# =============================================================================
def DSConv(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch),
        nn.Conv2d(in_ch, out_ch, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class LightSTNNN(nn.Module):
    def __init__(self, input_ch=4, output_ch=4, hidden=128, dropout_rate=0.1):
        super().__init__()
        from models import DeepConvLSTM
        self.enc = nn.Sequential(DSConv(input_ch, 64), DSConv(64, hidden), nn.Dropout2d(dropout_rate))
        self.conv_lstm = DeepConvLSTM(hidden, hidden, num_layers=2, kernel_size=3, padding=1)
        self.dec = nn.Sequential(DSConv(hidden, 64), nn.Conv2d(64, output_ch, 3, padding=1))
    def forward(self, x):
        B, S, C, H, W = x.shape
        z = self.enc(x.view(B*S, C, H, W)).view(B, S, -1, H, W)
        z, _ = self.conv_lstm(z)
        return self.dec(z[:, -1])

class LightSurrogateModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        from models import FiLMLayer
        self.stnnn = LightSTNNN(hidden=hidden)
        self.param_encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, 64))
        self.film = FiLMLayer(conditioning_dim=64, feature_channels=hidden)
    def normalize_params(self, ra, ha, q, da):
        ra_n = (torch.log10(torch.clamp(ra, min=1.0)) - 2.0) / 6.0
        ha_n = ha / 100.0
        q_n = (q + 10.0) / 20.0
        da_n = (torch.log10(torch.clamp(da, min=1e-5)) + 3.0) / 2.176
        return ra_n, ha_n, q_n, da_n
    def forward(self, x_seq, ra, ha, q, da):
        B, S, C, H, W = x_seq.shape
        p_n = self.normalize_params(ra, ha, q, da)
        p_vec = torch.stack(p_n, dim=-1)
        z_enc = self.stnnn.enc(x_seq.view(B*S, C, H, W)).view(B, S, -1, H, W)
        z_lstm, _ = self.stnnn.conv_lstm(z_enc)
        p_emb = self.param_encoder(p_vec)
        z_mod = self.film(z_lstm[:, -1], p_emb)
        return self.stnnn.dec(z_mod)

# =============================================================================
# 4. 훈련 루프 (v4 스케줄 완전 이식)
# =============================================================================
def train_fast(args, model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    
    sample_ds = train_loader.dataset.datasets[0] if hasattr(train_loader.dataset, 'datasets') else train_loader.dataset
    phys_params = {**sample_ds.params, 'norm_params': sample_ds.norm_params}
    phys = FastPhysicsLoss(phys_params, sample_ds.nano_props, 
                          dt=phys_params.get('dt', 0.0001), 
                          dx=1.0/(sample_ds.nx-1), dy=1.0/(sample_ds.ny-1)).to(device)

    # 1. 물리 손실 가중치 초기화
    norm_weights = calculate_physics_normalization(model, train_loader, phys, device)

    best_val_loss = float('inf')
    target_physics_lambda = 0.05
    warmup_threshold = int(args.epochs * 0.15)
    ramp_up_period = int(args.epochs * 0.10)

    for epoch in range(args.epochs):
        # v4 스케줄링 로직
        if epoch < warmup_threshold:
            current_phys_lambda = 0.0
            phase_str = "Data Warmup"
        else:
            ramp_weight = min(1.0, (epoch - warmup_threshold) / (ramp_up_period + 1e-8))
            current_phys_lambda = target_physics_lambda * ramp_weight
            phase_str = f"Phys Ramp ({ramp_weight*100:.0f}%)"

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [{phase_str}]")
        for inp, tgt, pd in pbar:
            inp, tgt = inp.to(device), tgt.to(device)
            r, h, q, d = [pd[k].to(device) for k in ['Ra', 'Ha', 'Q', 'Da']]
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred = model(inp, r, h, q, d)
                loss_mse = F.mse_loss(pred, tgt)
            
            if current_phys_lambda > 0:
                with torch.amp.autocast('cuda', enabled=False):
                    p_res = phys.physics_residual_loss(inp[:, -1].float(), pred.float(), r, h, q, d)
                    loss_phys = (p_res['continuity'].mean() * norm_weights['continuity'] + 
                                p_res['momentum_x'].mean() * norm_weights['momentum_x'] + 
                                p_res['momentum_y'].mean() * norm_weights['momentum_y'] + 
                                p_res['energy'].mean() * norm_weights['energy'])
            else:
                loss_phys = torch.tensor(0.0, device=device)
            
            loss_total = loss_mse + current_phys_lambda * loss_phys
            
            if torch.isfinite(loss_total):
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            
            pbar.set_postfix({'mse': f"{loss_mse.item():.2e}", 'phys': f"{loss_phys.item():.2e}"})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, t, p in val_loader:
                vp = model(i.to(device), p['Ra'].to(device), p['Ha'].to(device), p['Q'].to(device), p['Da'].to(device))
                val_loss += F.mse_loss(vp, t.to(device)).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"  Val MSE: {avg_val:.6f}")
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"light_model_{args.base_fluid}.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_fluid', default='EG')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_root', default='data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = os.path.join(args.data_root, args.base_fluid)
    all_files = [f for f in sorted(glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)) if 'phi' not in f.lower()]
    random.shuffle(all_files)
    
    cache_dir = f"cache_{args.base_fluid}"
    caches = [CachedSequenceDataset(cp) for f in tqdm(all_files[:200], desc="Caching") if (cp := preprocess_to_hdf5(f, cache_dir))]
    
    if not caches: return
    train_size = int(len(caches) * 0.8)
    loader_args = {'batch_size': args.batch_size, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2}
    train_loader = DataLoader(ConcatDataset(caches[:train_size]), shuffle=True, **loader_args)
    val_loader = DataLoader(ConcatDataset(caches[train_size:]), **loader_args)

    model = LightSurrogateModel(hidden=128).to(device)
    train_fast(args, model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
