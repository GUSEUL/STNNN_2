"""
Multi-dataset loader for handling multiple .mat files from different materials.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import numpy as np
from data import MatDataset

class MultiMatDataset(Dataset):
    """Dataset class for loading multiple .mat files and combining them."""
    
    def __init__(self, file_list_path, device='cpu'):
        """Initialize dataset from a list of .mat files.
        
        Args:
            file_list_path (str): Path to text file containing list of .mat files
            device (str): Device to store tensors on
        """
        self.device = device
        self.datasets = []
        self.cumulative_lengths = [0]
        self.file_paths = []
        
        # Read file list
        if not os.path.exists(file_list_path):
            raise FileNotFoundError(f"File list {file_list_path} not found")
            
        with open(file_list_path, 'r', encoding='utf-8') as f:
            file_paths = []
            for line in f:
                path = line.strip()
                # Remove BOM if present
                if path.startswith('\ufeff'):
                    path = path[1:]
                # Remove any other non-printable characters
                path = ''.join(char for char in path if char.isprintable() or char in [' ', '\t'])
                if path:
                    # Convert Unix-style paths to Windows format if needed
                    if path.startswith('/c/'):
                        path = 'C:' + path[2:].replace('/', '\\')
                    elif path.startswith('/'):
                        # Handle other Unix paths
                        path = path.replace('/', '\\')
                    # Normalize path separators
                    path = os.path.normpath(path)
                    file_paths.append(path)
        
        if not file_paths:
            raise ValueError("No files found in file list")
            
        print(f"Loading {len(file_paths)} datasets...")
        
        # Load each dataset
        total_length = 0
        for i, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            try:
                dataset = MatDataset(file_path, device=device)
                self.datasets.append(dataset)
                self.file_paths.append(file_path)
                total_length += len(dataset)
                self.cumulative_lengths.append(total_length)
                print(f"  {i+1}. Loaded {os.path.basename(file_path)} ({len(dataset)} samples)")
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        if not self.datasets:
            raise ValueError("No datasets could be loaded successfully")
            
        print(f"Successfully loaded {len(self.datasets)} datasets with {total_length} total samples")
        
        # Store parameters from the first dataset (assuming similar physics parameters)
        self.params = self.datasets[0].get_params()
        self.nanofluid_props = self.datasets[0].get_nanofluid_properties()
        
        # Grid dimensions (should be consistent across all datasets)
        self.ny = self.datasets[0].ny
        self.nx = self.datasets[0].nx
        
    def __len__(self):
        """Return total number of samples across all datasets."""
        return self.cumulative_lengths[-1]
        
    def __getitem__(self, idx):
        """Get data sample by finding the appropriate dataset and local index."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cum_len:
                dataset_idx = i - 1
                break
        
        # Calculate local index within the found dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        # Get sample from the appropriate dataset
        return self.datasets[dataset_idx][local_idx]
    
    def get_params(self):
        """Return physical and normalization parameters."""
        return self.params
    
    def get_nanofluid_properties(self):
        """Return nanofluid properties for physics loss initialization."""
        return self.nanofluid_props
    
    def get_dataset_info(self):
        """Return information about loaded datasets."""
        info = {
            'num_datasets': len(self.datasets),
            'total_samples': len(self),
            'file_paths': self.file_paths,
            'samples_per_dataset': [len(ds) for ds in self.datasets]
        }
        return info

def load_selected_files_dataset(selected_files_path, device='cpu'):
    """
    Load dataset from selected files list.
    
    Args:
        selected_files_path (str): Path to file containing selected .mat files
        device (str): Device to store tensors on
        
    Returns:
        MultiMatDataset: Combined dataset from all selected files
    """
    if not os.path.exists(selected_files_path):
        # Fallback to single file for backwards compatibility
        fallback_file = '12_Ra_38552.mat'
        print(f"Selected files list not found at {selected_files_path}")
        print(f"Using fallback single file: {fallback_file}")
        
        if os.path.exists(fallback_file):
            return MatDataset(fallback_file, device=device)
        else:
            raise FileNotFoundError(f"Neither selected files list nor fallback file found")
    
    return MultiMatDataset(selected_files_path, device=device)