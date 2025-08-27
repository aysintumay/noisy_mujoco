"""
Data utilities for time series experiment evaluation.
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import sys
import os
sys.path.append("..")

class TimeSeriesExperimentDataset(Dataset):
    """Dataset for time series experiments with proper data handling."""
    
    def __init__(self, data_all, input_horizon=6, output_horizon=6, drop_col_11=False):
        super().__init__()
        self.input_horizon = input_horizon
        self.output_horizon = output_horizon
        self.drop_col_11 = drop_col_11
        
        # Handle NaN column (index 11) by dropping it if requested
        if drop_col_11 and data_all.shape[2] > 11:
            # Drop column 11 (which has NaNs)
            mask = torch.ones(data_all.shape[2], dtype=bool)
            mask[11] = False
            data_all = data_all[:, :, mask]
        
        self.data_all = data_all
        self.data, self.pl, self.labels = self.prep_data()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure float32 dtype for all tensors
        return (self.data[idx].float(), 
                self.pl[idx].float(), 
                self.labels[idx].float())

    def prep_data(self):
        """Prepare data for training."""
        n = self.data_all.shape[0]
        
        # Input: first input_horizon time steps
        x = self.data_all[:, :self.input_horizon, :]
        
        # Control (p-level): last output_horizon time steps of last column
        pl = self.data_all[:, self.input_horizon:self.input_horizon + self.output_horizon, -1]
        
        # Target: last output_horizon time steps, all features except p-level
        y = self.data_all[:, self.input_horizon:self.input_horizon + self.output_horizon, :-1]
        y = y.reshape((n, -1))  # Flatten for training
        
        return x, pl, y

def load_and_normalize_data(data_path: str, drop_col_11: bool = False) -> Dict:
    """Load and normalize the time series data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get original shapes and statistics
    mean = data['mean']
    std = data['std']
    
    # Handle column 11 (NaN column) if needed
    if drop_col_11:
        # Drop column 11 from data and statistics
        mask = torch.ones(mean.shape[0], dtype=bool)
        mask[11] = False
        
        mean = mean[mask]
        std = std[mask]
        
        # Drop from all data splits
        train_data = data['train'][:, :, mask]
        val_data = data['val'][:, :, mask]
        test_data = data['test'][:, :, mask]
    else:
        train_data = data['train']
        val_data = data['val'] 
        test_data = data['test']
    
    # Normalize data and ensure float32 dtype
    normalized_train = ((train_data - mean) / std).float()
    normalized_val = ((val_data - mean) / std).float()
    normalized_test = ((test_data - mean) / std).float()
    
    return {
        'train': normalized_train,
        'val': normalized_val,
        'test': normalized_test,
        'mean': mean,
        'std': std,
        'raw_train': train_data,
        'raw_val': val_data,
        'raw_test': test_data
    }

def create_datasets(normalized_data: Dict, input_horizon: int = 6, 
                   output_horizon: int = 6, drop_col_11: bool = False) -> Tuple:
    """Create train/val/test datasets."""
    
    train_dataset = TimeSeriesExperimentDataset(
        normalized_data['train'], input_horizon, output_horizon, drop_col_11=False  # Already handled
    )
    val_dataset = TimeSeriesExperimentDataset(
        normalized_data['val'], input_horizon, output_horizon, drop_col_11=False
    )
    test_dataset = TimeSeriesExperimentDataset(
        normalized_data['test'], input_horizon, output_horizon, drop_col_11=False
    )
    
    return train_dataset, val_dataset, test_dataset

def get_static_dynamic_splits(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split data into static and dynamic p-level samples.
    
    Args:
        data: Raw data tensor of shape [batch, time, features]
        
    Returns:
        static_mask: Boolean mask for static p-level samples
        dynamic_mask: Boolean mask for dynamic p-level samples
    """
    # P-level is the last column
    p_levels = data[:, :, -1]  # [batch, time]
    
    # Check if p-level changes over time for each sample
    # Static: no change across all time steps
    p_level_changes = torch.abs(p_levels[:, 1:] - p_levels[:, :-1])  # [batch, time-1]
    max_change = torch.max(p_level_changes, dim=1)[0]  # [batch]
    
    static_mask = max_change < 1e-6  # Very small threshold for floating point
    dynamic_mask = ~static_mask
    
    return static_mask, dynamic_mask

def calculate_trend(map_values: torch.Tensor) -> torch.Tensor:
    """Calculate trend classification for MAP values.
    
    Args:
        map_values: MAP values of shape [batch, time_steps]
        
    Returns:
        trends: Trend classification (0=down, 1=flat, 2=up) of shape [batch]
    """
    # Calculate slope over time steps
    time_steps = torch.arange(map_values.shape[1], dtype=torch.float32)
    
    # For each sample, calculate linear regression slope
    slopes = []
    for i in range(map_values.shape[0]):
        y = map_values[i]
        # Simple slope calculation: (y_end - y_start) / (t_end - t_start)
        slope = (y[-1] - y[0]) / (time_steps[-1] - time_steps[0])
        slopes.append(slope)
    
    slopes = torch.tensor(slopes)
    
    # Classify trends
    trends = torch.ones_like(slopes, dtype=torch.long)  # Default to flat (1)
    trends[slopes > 1.0] = 2  # Up
    trends[slopes < -1.0] = 0  # Down
    
    return trends

if __name__ == "__main__":
    # Test the data loading
    data_path = '/abiomed/downsampled/10min_1hr_all_data.pkl'
    
    print("Testing data loading...")
    normalized_data = load_and_normalize_data(data_path, drop_col_11=True)
    
    print(f"Original shape: {normalized_data['train'].shape}")
    print(f"Mean shape: {normalized_data['mean'].shape}")
    print(f"Std shape: {normalized_data['std'].shape}")
    
    train_dataset, val_dataset, test_dataset = create_datasets(normalized_data)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test a sample
    x, pl, y = train_dataset[0]
    print(f"Sample x shape: {x.shape}")
    print(f"Sample pl shape: {pl.shape}")
    print(f"Sample y shape: {y.shape}")
    
    # Test static/dynamic split
    static_mask, dynamic_mask = get_static_dynamic_splits(normalized_data['raw_test'])
    print(f"Static samples: {static_mask.sum().item()}")
    print(f"Dynamic samples: {dynamic_mask.sum().item()}")