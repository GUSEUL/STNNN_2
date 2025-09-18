"""
Dataset loader for PhyCRNet.
Handles loading and preprocessing of .mat data files.
"""

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available. MATLAB v7.3 files will not be supported.")

def load_mat_file(mat_file_path):
    """
    Load MATLAB file with support for both v7.0 and v7.3 formats.
    
    Args:
        mat_file_path (str): Path to .mat file
        
    Returns:
        dict: Dictionary containing all variables from the file
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MATLAB file {mat_file_path} not found")
    
    try:
        # Try scipy.io first (for v7.0 and earlier)
        mat_data = sio.loadmat(mat_file_path)
        print(f"Loaded {mat_file_path} with scipy.io (MATLAB v7.0 or earlier)")
        return mat_data
    except NotImplementedError:
        if not HAS_H5PY:
            raise ImportError("MATLAB v7.3 file detected but h5py is not available. Please install h5py: pip install h5py")
        
        print(f"Loading {mat_file_path} with h5py (MATLAB v7.3)")
        # Load with h5py for v7.3 files
        with h5py.File(mat_file_path, 'r') as f:
            mat_data = {}
            for key in f.keys():
                if not key.startswith('#'):  # Skip h5py metadata
                    mat_data[key] = np.array(f[key])
        return mat_data

def extract_nanofluid_properties(mat_data):
    """
    Extract nanofluid properties from loaded MATLAB data.
    
    Args:
        mat_data (dict): Dictionary from load_mat_file()
        
    Returns:
        dict: Dictionary containing nanofluid properties
    """
    def extract_scalar(mat_var, default_value, var_name):
        """Safely extract scalar value from MATLAB variable."""
        try:
            if mat_var is None:
                print(f"Warning: {var_name} not found, using default value {default_value}")
                return default_value
            
            # Handle numpy arrays (common case)
            if hasattr(mat_var, 'shape'):
                if mat_var.size == 1:
                    return float(mat_var.item())  # Extract single value safely
                else:
                    print(f"Warning: {var_name} has multiple values, using first element")
                    return float(mat_var.flat[0])
            
            # Handle direct scalar values
            return float(mat_var)
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Warning: Could not extract {var_name}, using default value {default_value}. Error: {e}")
            return default_value
    
    # Extract nanofluid properties - these names match the actual MATLAB file
    nanofluid_props = {
        # Kinematic viscosity
        'nu_thnf': extract_scalar(mat_data.get('nuthnf'), 1.0, 'nuthnf'),
        'nu_f': extract_scalar(mat_data.get('nuf'), 1.0, 'nuf'),
        
        # Electrical conductivity
        'sigma_thnf': extract_scalar(mat_data.get('sigthnf'), 1.0, 'sigthnf'),
        'sigma_f': extract_scalar(mat_data.get('sigf'), 1.0, 'sigf'),
        
        # Density
        'rho_thnf': extract_scalar(mat_data.get('rothnf'), 1000.0, 'rothnf'),
        'rho_f': extract_scalar(mat_data.get('rof'), 1000.0, 'rof'),
        
        # Thermal expansion coefficient
        'beta_thnf': extract_scalar(mat_data.get('bethnf'), 1.0, 'bethnf'),
        'beta_f': extract_scalar(mat_data.get('bef'), 1.0, 'bef'),
        
        # Thermal diffusivity
        'alpha_thnf': extract_scalar(mat_data.get('althnf'), 1.0, 'althnf'),
        'alpha_f': extract_scalar(mat_data.get('alf'), 1.0, 'alf'),
        
        # Heat capacity
        'rhocp_thnf': extract_scalar(mat_data.get('rocpthnf'), 1.0, 'rocpthnf'),
        'rhocp_f': extract_scalar(mat_data.get('rocpf'), 1.0, 'rocpf'),
    }
    
    # Calculate ratios
    ratios = {
        'nu_thnf_ratio': nanofluid_props['nu_thnf'] / nanofluid_props['nu_f'],
        'sigma_thnf_ratio': nanofluid_props['sigma_thnf'] / nanofluid_props['sigma_f'],
        'rho_f_thnf_ratio': nanofluid_props['rho_f'] / nanofluid_props['rho_thnf'],
        'beta_thnf_ratio': nanofluid_props['beta_thnf'] / nanofluid_props['beta_f'],
        'alpha_thnf_ratio': nanofluid_props['alpha_thnf'] / nanofluid_props['alpha_f'],
        'rhocp_f_thnf_ratio': nanofluid_props['rhocp_f'] / nanofluid_props['rhocp_thnf'],
    }
    
    print(f"Nanofluid properties loaded:")
    print(f"  ν_thnf/ν_f: {ratios['nu_thnf_ratio']:.6f}")
    print(f"  σ_thnf/σ_f: {ratios['sigma_thnf_ratio']:.6f}")
    print(f"  ρ_f/ρ_thnf: {ratios['rho_f_thnf_ratio']:.6f}")
    print(f"  β_thnf/β_f: {ratios['beta_thnf_ratio']:.6f}")
    print(f"  α_thnf/α_f: {ratios['alpha_thnf_ratio']:.6f}")
    print(f"  (ρC_p)_f/(ρC_p)_thnf: {ratios['rhocp_f_thnf_ratio']:.6f}")
    
    return {**nanofluid_props, **ratios}

class MatDataset(Dataset):
    """Dataset class for loading .mat files with MAC grid data.
    
    The data is stored in a staggered MAC grid format with time as first dimension:
    - u: (time, 32, 31) staggered in x
    - v: (time, 31, 32) staggered in y
    - p: (time, 32, 32) cell-centered
    - θ: (time, 32, 32) cell-centered
    """
    
    def __init__(self, matfile, device='cpu', time_slice_end=None):
        """Initialize dataset from .mat file.
        
        Args:
            matfile (str): Path to .mat data file
            device (str): Device to store tensors on
            time_slice_end (float, optional): If provided, slices the data to this end time. Defaults to None.
        """
        # Load MATLAB file with v7.3 support
        d = load_mat_file(matfile)
        
        # Extract nanofluid properties
        self.nanofluid_props = extract_nanofluid_properties(d)
        
        # Extract data arrays - Time is already the first dimension
        u_raw = d['ustore']   # (time, 32, 31)  staggered in x
        v_raw = d['vstore']   # (time, 31, 32)  staggered in y  
        p_raw = d['pstore']   # (time, 32, 32)  cell-centered
        t_raw = d['tstore']   # (time, 32, 32)  cell-centered

        # Store physical parameters with safe extraction
        def safe_extract_param(key, default_value):
            """Safely extract parameter with default fallback."""
            try:
                return float(d[key].squeeze()) if key in d else default_value
            except:
                return default_value
        
        self.params = {
            'Ra': safe_extract_param('Ra', 1e4),     # Rayleigh number
            'Ha': safe_extract_param('Ha', 0.0),     # Hartmann number
            'Pr': safe_extract_param('Pr', 0.71),    # Prandtl number
            'Da': safe_extract_param('Da', 1e-3),    # Darcy number
            'Q': safe_extract_param('Q', 0.0),       # Heat source/sink parameter
            'dt': safe_extract_param('dt', 0.0001),  # Time step
        }

        # --- Time Slicing Logic ---
        if time_slice_end is not None:
            dt = self.params.get('dt', 0.0001)
            num_timesteps = u_raw.shape[0]
            time_vector = np.arange(num_timesteps) * dt
            
            # Find the index up to which we should keep the data
            slice_idx = np.searchsorted(time_vector, time_slice_end, side='right')
            
            if slice_idx > 0:
                print(f"\n--- Slicing data up to time ~{time_slice_end}s (index: {slice_idx}) ---")
                u_raw = u_raw[:slice_idx]
                v_raw = v_raw[:slice_idx]
                p_raw = p_raw[:slice_idx]
                t_raw = t_raw[:slice_idx]
                print(f"New data shape (time, y, x): {u_raw.shape}")
            else:
                print(f"Warning: time_slice_end ({time_slice_end}) is too small. Using full dataset.")
        
        print(f"\nLoaded data shapes:")
        print(f"  u: {u_raw.shape} (time, y, x)")
        print(f"  v: {v_raw.shape} (time, y, x)")
        print(f"  p: {p_raw.shape} (time, y, x)")
        print(f"  t: {t_raw.shape} (time, y, x)")
        
        # Transpose from (time, y, x) to (y, x, time) for compatibility with existing code
        u_raw = np.transpose(u_raw, (1, 2, 0))
        v_raw = np.transpose(v_raw, (1, 2, 0))
        p_raw = np.transpose(p_raw, (1, 2, 0))
        t_raw = np.transpose(t_raw, (1, 2, 0))
        
        print(f"\nAfter transpose to (y, x, time):")
        print(f"  u: {u_raw.shape}")
        print(f"  v: {v_raw.shape}")
        print(f"  p: {p_raw.shape}")
        print(f"  t: {t_raw.shape}")
        
        # Calculate normalization factors
        self.u_mean, self.u_std = np.mean(u_raw), np.std(u_raw)
        self.v_mean, self.v_std = np.mean(v_raw), np.std(v_raw)
        self.p_mean, self.p_std = np.mean(p_raw), np.std(p_raw)
        self.t_mean, self.t_std = np.mean(t_raw), np.std(t_raw)
        
        # Normalize data
        u_norm = (u_raw - self.u_mean) / (self.u_std + 1e-8)
        v_norm = (v_raw - self.v_mean) / (self.v_std + 1e-8)
        p_norm = (p_raw - self.p_mean) / (self.p_std + 1e-8)
        t_norm = (t_raw - self.t_mean) / (self.t_std + 1e-8)
        
        # Convert to PyTorch tensors
        self.u = torch.tensor(u_norm, dtype=torch.float32, device=device)
        self.v = torch.tensor(v_norm, dtype=torch.float32, device=device)
        self.p = torch.tensor(p_norm, dtype=torch.float32, device=device)
        self.t = torch.tensor(t_norm, dtype=torch.float32, device=device)
        
        # Store normalization parameters as tensors
        self.norm_params = {
            'u': (torch.tensor(self.u_mean, device=device), torch.tensor(self.u_std, device=device)),
            'v': (torch.tensor(self.v_mean, device=device), torch.tensor(self.v_std, device=device)),
            'p': (torch.tensor(self.p_mean, device=device), torch.tensor(self.p_std, device=device)),
            't': (torch.tensor(self.t_mean, device=device), torch.tensor(self.t_std, device=device))
        }
        
        # Grid dimensions from pressure/temperature field
        self.ny, self.nx = self.p.shape[0], self.p.shape[1]
        self.nt = self.p.shape[2]
        
        # Number of usable time-steps (total - 1)
        self.T = self.nt - 1
        
    def __len__(self):
        """Return number of time steps available for training."""
        return self.T
        
    def __getitem__(self, idx):
        """Get data for time steps idx and idx+1.
        
        Args:
            idx (int): Time index
            
        Returns:
            tuple: (Current state, Next state, Ground truth dict)
                - States are cell-centered [4×H×W] tensors
                - Ground truth is MAC grid format dictionary
        """
        # Convert staggered velocities to cell-centered at time idx
        u_c = 0.5*(self.u[:, :-1, idx] + self.u[:, 1:, idx])
        v_c = 0.5*(self.v[:-1, :, idx] + self.v[1:, :, idx])
        
        # Combine into cell-centered state at time idx
        f0 = torch.zeros(4, self.ny, self.nx, device=self.p.device)
        f0[0, :u_c.shape[0], :u_c.shape[1]] = u_c  # U
        f0[1, :v_c.shape[0], :v_c.shape[1]] = v_c  # V
        f0[2] = self.t[:,:,idx]  # θ
        f0[3] = self.p[:,:,idx]  # P
        
        # Same procedure for time idx+1
        u1_c = 0.5*(self.u[:, :-1, idx+1] + self.u[:, 1:, idx+1])
        v1_c = 0.5*(self.v[:-1, :, idx+1] + self.v[1:, :, idx+1])
        
        f1 = torch.zeros(4, self.ny, self.nx, device=self.p.device)
        f1[0, :u1_c.shape[0], :u1_c.shape[1]] = u1_c  # U
        f1[1, :v1_c.shape[0], :v1_c.shape[1]] = v1_c  # V
        f1[2] = self.t[:,:,idx+1]  # θ
        f1[3] = self.p[:,:,idx+1]  # P
        
        # Ground truth in original MAC staggered format
        gt = {
            'u': self.u[:,:,idx+1].unsqueeze(0),  # [1×H×W-1]
            'v': self.v[:,:,idx+1].unsqueeze(0),  # [1×H-1×W]
            'p': self.p[:,:,idx+1].unsqueeze(0),  # [1×H×W]
            't': self.t[:,:,idx+1].unsqueeze(0)   # [1×H×W]
        }
        
        return f0, f1, gt
    
    def get_params(self):
        """Return physical and normalization parameters"""
        return {
            **self.params,
            'norm_params': self.norm_params,
            'nanofluid_props': self.nanofluid_props
        }
    
    def get_nanofluid_properties(self):
        """Return nanofluid properties for physics loss initialization"""
        return self.nanofluid_props 