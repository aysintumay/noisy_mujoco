"""
Comprehensive evaluation metrics for time series forecasting.
"""
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.stats import rankdata
import warnings

def unnormalize_predictions(predictions: torch.Tensor, targets: torch.Tensor, 
                          mean: torch.Tensor, std: torch.Tensor, 
                          forecast_horizon: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unnormalize predictions and targets.
    
    Args:
        predictions: Normalized predictions [batch, forecast_horizon * features]
        targets: Normalized targets [batch, forecast_horizon * features]
        mean: Feature means [features-1] (excluding p-level)
        std: Feature stds [features-1] (excluding p-level)
        forecast_horizon: Number of forecast time steps
        
    Returns:
        unnorm_predictions, unnorm_targets: Unnormalized tensors
    """
    # Reshape to [batch, forecast_horizon, features]
    batch_size = predictions.shape[0]
    num_features = predictions.shape[1] // forecast_horizon
    
    pred_reshaped = predictions.view(batch_size, forecast_horizon, num_features)
    target_reshaped = targets.view(batch_size, forecast_horizon, num_features)
    
    # Unnormalize
    unnorm_pred = pred_reshaped * std[:num_features] + mean[:num_features]
    unnorm_target = target_reshaped * std[:num_features] + mean[:num_features]
    
    return unnorm_pred, unnorm_target

def mae_all_features(predictions: torch.Tensor, targets: torch.Tensor, 
                    mean: torch.Tensor, std: torch.Tensor, 
                    forecast_horizon: int = 6) -> float:
    """Calculate MAE averaged across all features and time steps.
    
    Args:
        predictions: Model predictions [batch, forecast_horizon * features]
        targets: Ground truth [batch, forecast_horizon * features]
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        forecast_horizon: Number of forecast time steps
        
    Returns:
        mae: Mean absolute error across all features and time steps
    """
    unnorm_pred, unnorm_target = unnormalize_predictions(
        predictions, targets, mean, std, forecast_horizon
    )
    
    mae = torch.mean(torch.abs(unnorm_pred - unnorm_target)).item()
    return mae

def mae_map_only(predictions: torch.Tensor, targets: torch.Tensor, 
                mean: torch.Tensor, std: torch.Tensor, 
                forecast_horizon: int = 6) -> float:
    """Calculate MAE for MAP (feature index 0) only.
    
    Args:
        predictions: Model predictions [batch, forecast_horizon * features]
        targets: Ground truth [batch, forecast_horizon * features]
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        forecast_horizon: Number of forecast time steps
        
    Returns:
        mae_map: Mean absolute error for MAP feature only
    """
    unnorm_pred, unnorm_target = unnormalize_predictions(
        predictions, targets, mean, std, forecast_horizon
    )
    
    # Extract MAP (first feature)
    map_pred = unnorm_pred[:, :, 0]  # [batch, forecast_horizon]
    map_target = unnorm_target[:, :, 0]  # [batch, forecast_horizon]
    
    mae_map = torch.mean(torch.abs(map_pred - map_target)).item()
    return mae_map

def mae_static_dynamic_split(predictions: torch.Tensor, targets: torch.Tensor,
                           raw_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                           forecast_horizon: int = 6) -> Tuple[float, float]:
    """Calculate MAE for static vs dynamic p-level samples.
    
    Args:
        predictions: Model predictions [batch, forecast_horizon * features]
        targets: Ground truth [batch, forecast_horizon * features]
        raw_data: Raw data for determining static/dynamic [batch, total_time, all_features]
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        forecast_horizon: Number of forecast time steps
        
    Returns:
        mae_static, mae_dynamic: MAE for static and dynamic samples
    """
    from data_utils import get_static_dynamic_splits
    
    # Get static/dynamic masks
    static_mask, dynamic_mask = get_static_dynamic_splits(raw_data)
    
    if static_mask.sum() == 0:
        mae_static = float('nan')
    else:
        static_pred = predictions[static_mask]
        static_target = targets[static_mask]
        mae_static = mae_all_features(static_pred, static_target, mean, std, forecast_horizon)
    
    if dynamic_mask.sum() == 0:
        mae_dynamic = float('nan')
    else:
        dynamic_pred = predictions[dynamic_mask]
        dynamic_target = targets[dynamic_mask]
        mae_dynamic = mae_all_features(dynamic_pred, dynamic_target, mean, std, forecast_horizon)
    
    return mae_static, mae_dynamic

def trend_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                  mean: torch.Tensor, std: torch.Tensor,
                  forecast_horizon: int = 6) -> float:
    """Calculate trend accuracy for MAP values.
    
    Args:
        predictions: Model predictions [batch, forecast_horizon * features]
        targets: Ground truth [batch, forecast_horizon * features]
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        forecast_horizon: Number of forecast time steps
        
    Returns:
        accuracy: Trend classification accuracy
    """
    from data_utils import calculate_trend
    
    unnorm_pred, unnorm_target = unnormalize_predictions(
        predictions, targets, mean, std, forecast_horizon
    )
    
    # Extract MAP values
    map_pred = unnorm_pred[:, :, 0]  # [batch, forecast_horizon]
    map_target = unnorm_target[:, :, 0]  # [batch, forecast_horizon]
    
    # Calculate trends
    pred_trends = calculate_trend(map_pred)
    target_trends = calculate_trend(map_target)
    
    # Calculate accuracy
    correct = (pred_trends == target_trends).float()
    accuracy = torch.mean(correct).item()
    
    return accuracy

def crps_score(predictions: torch.Tensor, targets: torch.Tensor, 
              samples: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
              forecast_horizon: int = 6) -> float:
    """Calculate CRPS (Continuous Ranked Probability Score).
    
    Args:
        predictions: Point predictions [batch, forecast_horizon * features] (not used but kept for interface)
        targets: Ground truth [batch, forecast_horizon * features]
        samples: Prediction samples [num_samples, batch, forecast_horizon * features]
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        forecast_horizon: Number of forecast time steps
        
    Returns:
        crps: CRPS score
    """
    num_samples = samples.shape[0]
    batch_size = samples.shape[1]
    
    # Unnormalize samples and targets
    crps_values = []
    
    for b in range(batch_size):
        sample_batch = samples[:, b:b+1, :]  # [num_samples, 1, features]
        target_batch = targets[b:b+1, :]  # [1, features]
        
        # Reshape samples properly for unnormalization
        # sample_batch is [num_samples, 1, features] -> flatten each sample individually
        unnorm_samples_list = []
        for i in range(num_samples):
            single_sample = sample_batch[i:i+1, 0, :]  # [1, features]
            unnorm_sample, _ = unnormalize_predictions(
                single_sample, target_batch, mean, std, forecast_horizon
            )
            unnorm_samples_list.append(unnorm_sample)
        
        # Stack back to [num_samples, forecast_horizon, features]
        unnorm_samples = torch.stack(unnorm_samples_list, dim=0)  # [num_samples, 1, forecast_horizon, features]
        unnorm_samples = unnorm_samples.squeeze(1)  # [num_samples, forecast_horizon, features]
        
        # Unnormalize target
        _, unnorm_target = unnormalize_predictions(
            target_batch, target_batch, mean, std, forecast_horizon
        )
        
        # Calculate CRPS for this sample
        sample_crps = _calculate_sample_crps(unnorm_samples, unnorm_target)
        crps_values.append(sample_crps)
    
    return np.mean(crps_values)

def _calculate_sample_crps(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate CRPS for a single sample."""
    samples_np = samples.detach().cpu().numpy()  # [num_samples, forecast_horizon, features]
    target_np = target.detach().cpu().numpy()  # [1, forecast_horizon, features]
    
    # Calculate CRPS for each feature and time step
    crps_values = []
    
    for t in range(samples_np.shape[1]):
        for f in range(samples_np.shape[2]):
            sample_values = samples_np[:, t, f]
            target_value = target_np[0, t, f]
            
            # CRPS calculation
            crps_val = _crps_empirical(sample_values, target_value)
            crps_values.append(crps_val)
    
    return np.mean(crps_values)

def _crps_empirical(samples: np.ndarray, target: float) -> float:
    """Calculate empirical CRPS."""
    samples = np.sort(samples)
    n = len(samples)
    
    # Empirical CDF at target
    target_rank = np.searchsorted(samples, target, side='right')
    target_cdf = target_rank / n
    
    # CRPS calculation
    crps = 0.0
    
    # Integrate over all possible values
    for i in range(n):
        # CDF value at samples[i]
        cdf_i = (i + 1) / n
        
        if samples[i] <= target:
            crps += (cdf_i ** 2) * (samples[i+1] - samples[i] if i < n-1 else 0)
        else:
            crps += ((cdf_i - 1) ** 2) * (samples[i+1] - samples[i] if i < n-1 else 0)
    
    # Add contribution from target point
    if target_rank < n:
        if target_rank > 0:
            crps += (target_cdf ** 2) * (target - samples[target_rank - 1])
        crps += ((target_cdf - 1) ** 2) * (samples[target_rank] - target)
    
    return crps

def evaluate_model(model, test_loader, mean: torch.Tensor, std: torch.Tensor,
                  raw_test_data: torch.Tensor, forecast_horizon: int = 6,
                  num_samples: int = 50, device: str = 'cpu') -> Dict[str, float]:
    """Comprehensive evaluation of a model.
    
    Args:
        model: Trained model with forward() and sample_multiple() methods
        test_loader: DataLoader for test data
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        raw_test_data: Raw test data for static/dynamic split
        forecast_horizon: Number of forecast time steps
        num_samples: Number of samples for CRPS calculation
        device: Device to run evaluation on
        
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_samples = []
    
    print(f"Evaluating model on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (x, pl, y) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(test_loader)}")
                
            # Ensure consistent float32 dtype
            x = x.to(device, dtype=torch.float32)
            pl = pl.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Point predictions
            pred = model(x, pl)
            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())
            
            # Sample predictions for CRPS
            if hasattr(model, 'sample_multiple'):
                samples = model.sample_multiple(x, pl, num_samples=num_samples)
                all_samples.append(samples.cpu())
            else:
                # If no sampling method, repeat point prediction
                samples = pred.unsqueeze(0).repeat(num_samples, 1, 1)
                all_samples.append(samples.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_samples = torch.cat(all_samples, dim=1)  # [num_samples, total_batch, features]
    
    print("Calculating metrics...")
    
    # Calculate all metrics
    metrics = {}
    
    # 1. MAE all features
    metrics['mae_all_features'] = mae_all_features(
        all_predictions, all_targets, mean, std, forecast_horizon
    )
    
    # 2. MAP only MAE
    metrics['mae_map_only'] = mae_map_only(
        all_predictions, all_targets, mean, std, forecast_horizon
    )
    
    # 3. Static/Dynamic MAE
    mae_static, mae_dynamic = mae_static_dynamic_split(
        all_predictions, all_targets, raw_test_data, mean, std, forecast_horizon
    )
    metrics['mae_static'] = mae_static
    metrics['mae_dynamic'] = mae_dynamic
    
    # 4. Trend accuracy
    metrics['trend_accuracy'] = trend_accuracy(
        all_predictions, all_targets, mean, std, forecast_horizon
    )
    
    # 5. CRPS
    try:
        metrics['crps'] = crps_score(
            all_predictions, all_targets, all_samples, mean, std, forecast_horizon
        )
    except Exception as e:
        print(f"Warning: CRPS calculation failed: {e}")
        metrics['crps'] = float('nan')
    
    return metrics

if __name__ == "__main__":
    # Test the evaluation metrics
    print("Testing evaluation metrics...")
    
    batch_size = 100
    forecast_horizon = 6
    num_features = 11  # After dropping column 11
    output_dim = forecast_horizon * num_features
    
    # Create dummy data
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    mean = torch.randn(num_features)
    std = torch.ones(num_features)
    raw_data = torch.randn(batch_size, 12, 13)  # Full raw data with p-levels
    
    # Test individual metrics
    mae_all = mae_all_features(predictions, targets, mean, std, forecast_horizon)
    print(f"MAE all features: {mae_all:.4f}")
    
    mae_map = mae_map_only(predictions, targets, mean, std, forecast_horizon)
    print(f"MAE MAP only: {mae_map:.4f}")
    
    mae_static, mae_dynamic = mae_static_dynamic_split(
        predictions, targets, raw_data, mean, std, forecast_horizon
    )
    print(f"MAE static: {mae_static:.4f}, MAE dynamic: {mae_dynamic:.4f}")
    
    trend_acc = trend_accuracy(predictions, targets, mean, std, forecast_horizon)
    print(f"Trend accuracy: {trend_acc:.4f}")
    
    # Test CRPS with samples
    num_samples = 10
    samples = torch.randn(num_samples, batch_size, output_dim)
    crps = crps_score(predictions, targets, samples, mean, std, forecast_horizon)
    print(f"CRPS: {crps:.4f}")
    
    print("All metric tests passed!")