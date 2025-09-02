import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append("..")
from data_utils import load_and_normalize_data, create_datasets
from baselines import MLPDropoutBaseline, NeuralProcessBaseline, CLMUBaseline, StateSpaceBaseline
from evaluation_metrics import unnormalize_predictions
from model import WorldModel

def plot_output(env, pred, pl):
    """Original plot function - kept for backwards compatibility."""
    
    # unnorm x
    pred = torch.tensor(pred)

    std = env.world_model.std
    mean = env.world_model.mean
    columns = env.world_model.columns
    x_unnorm = pred.cpu() * std[columns] + mean[columns]

    total_len =  pred.shape[1] * pred.shape[0]

    fig, ax1 = plt.subplots()

    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/6:.1f}"))

    color = "tab:red"
    ax1.set_xlabel("time (hr)")
    ax1.set_ylabel("MAP", color=color)
    # multiple samples
    x_unnorm = x_unnorm.reshape(-1, 12)
    ax1.plot(
                np.arange(0, total_len, 1),
                x_unnorm[:, 0],
                color=color,
                linewidth=0.5,
                alpha=0.5,
    )
    ax1.lines[-1].set_label("prediction")

    ax1.legend()
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "P-Level", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(np.arange(0, total_len, 1), x_unnorm[:, 11], color=color, linewidth=1, alpha=0.5)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 10)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def load_baseline_model(model_path: str, model_type: str, device: str, 
                       input_dim: int, output_dim: int, normalized_data: Dict = None) -> torch.nn.Module:
    """Load a trained baseline model."""
    
    if model_type == 'mlp_dropout':
        model = MLPDropoutBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.2,
            device=device
        )
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    elif model_type == 'neural_process':
        model = NeuralProcessBaseline(
            input_dim=12,  # num_features
            output_dim=output_dim,
            latent_dim=128,
            hidden_dim=256,
            device=device
        )
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    elif model_type == 'clmu':
        model = CLMUBaseline(
            input_dim=12,  # num_features
            output_dim=output_dim,
            memory_dim=64,
            hidden_dim=128,
            num_layers=2,
            device=device
        )
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    elif model_type == 'state_space':
        model = StateSpaceBaseline(
            input_dim=12,  # num_features
            output_dim=output_dim,
            state_dim=64,
            hidden_dim=128,
            device=device
        )
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    elif model_type == 'rotary_transformer':
        # Create transformer model
        transformer = WorldModel(
            num_features=12,
            dim_model=256,
            num_heads=8,
            num_encoder_layers=3,
            num_decoder_layers=2,
            encoder_dropout=0.1,
            decoder_dropout=0.0,
            max_len=100,
            forecast_horizon=6,
            model_type='rotary_transformer',
            device=device
        )
        # Load model weights
        transformer.load_model(model_path)
        if normalized_data:
            transformer.mean = normalized_data['mean']
            transformer.std = normalized_data['std']
        return transformer.model
    elif model_type == 'regular_transformer':
        # Create transformer model
        transformer = WorldModel(
            num_features=12,
            dim_model=256,
            num_heads=8,
            num_encoder_layers=3,
            num_decoder_layers=2,
            encoder_dropout=0.1,
            decoder_dropout=0.0,
            max_len=100,
            forecast_horizon=6,
            model_type='transformer',
            device=device
        )
        # Load model weights
        transformer.load_model(model_path)
        if normalized_data:
            transformer.mean = normalized_data['mean']
            transformer.std = normalized_data['std']
        return transformer.model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def generate_baseline_predictions(models_dict: Dict[str, torch.nn.Module], 
                                data_loader, device: str, 
                                num_samples: int = None,
                                sample_indices: List[int] = None,
                                num_mc_samples: int = 50) -> Dict[str, Dict]:
    """Generate predictions from multiple baseline models for visualization.
    
    Args:
        models_dict: Dictionary of loaded models
        data_loader: DataLoader for samples
        device: Device to run inference on
        num_samples: Number of data samples to process
        sample_indices: Specific sample indices to process
        num_mc_samples: Number of Monte Carlo samples per model for uncertainty estimation
    """
    
    results = {}
    
    # Get a few samples for visualization
    sample_data = []
    sample_targets = []
    sample_inputs = []
    sample_pl = []
    
    with torch.no_grad():
        for i, (inputs, pl, targets) in enumerate(data_loader):
            
            if num_samples is not None and i >= num_samples:
                break
            if sample_indices is not None and i not in sample_indices:
                continue
            
            inputs = inputs.to(device)
            pl = pl.to(device)
            targets = targets.to(device)
            
            sample_inputs.append(inputs)
            sample_pl.append(pl)
            sample_targets.append(targets)
            
            batch_predictions = {}
            
            # Generate multiple predictions for each model for uncertainty estimation
            for model_name, model in models_dict.items():
                mc_predictions = []
                
                # for mc_idx in range(num_mc_samples):
                #     # Enable dropout/stochasticity during inference for some models
                #     if hasattr(model, 'train'):
                #         if model_name in ['mlp_dropout']:
                #             model.train()  # Enable dropout for uncertainty
                #         else:
                #             model.eval()   # Keep deterministic models in eval mode
                    
                #     pred = model(inputs, pl)
                #     mc_predictions.append(pred.cpu())
                
                 # Sample predictions for CRPS
                if hasattr(model, 'sample_multiple'):
                    samples = model.sample_multiple(inputs, pl, num_samples=num_mc_samples)
                    mc_predictions = samples.cpu()
                else:
                    # If no sampling method, repeat point prediction
                    samples = pred.unsqueeze(0).repeat(num_mc_samples, 1, 1)
                    mc_predictions = samples.cpu()
                # Stack predictions: [num_mc_samples, batch_size, features]
                batch_predictions[model_name] = mc_predictions
            
            sample_data.append(batch_predictions)
    
    return {
        'predictions': sample_data,
        'targets': sample_targets,
        'inputs': sample_inputs,
        'pl': sample_pl
    }

def plot_baseline_comparison(predictions_data: Dict, mean: torch.Tensor, std: torch.Tensor,
                           sample_idx: int = 0, forecast_horizon: int = 6,
                           save_path: Optional[str] = None, show_plot: bool = True,
                           uncertainty_alpha: float = 0.3, confidence_level: float = 0.9):
    """Plot ground truth, p-level, and probabilistic predictions for multiple baselines side by side.
    
    Args:
        predictions_data: Dict containing predictions, targets, inputs, and pl
        mean: Feature means for unnormalization  
        std: Feature stds for unnormalization
        sample_idx: Which sample to visualize
        forecast_horizon: Number of forecast steps
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        uncertainty_alpha: Alpha (transparency) for uncertainty bands
        confidence_level: Confidence level for uncertainty bands (e.g., 0.9 for 90%)
    """
    
    if sample_idx >= len(predictions_data['predictions']):
        raise ValueError(f"Sample index {sample_idx} out of range")
    
    # Get data for the specified sample
    sample_predictions = predictions_data['predictions'][sample_idx]
    sample_target = predictions_data['targets'][sample_idx][0]  # First item in batch
    sample_input = predictions_data['inputs'][sample_idx][0]   # First item in batch  
    sample_pl = predictions_data['pl'][sample_idx][0]         # First item in batch
    
    # Move tensors to CPU if needed
    if hasattr(sample_target, 'cpu'):
        sample_target = sample_target.cpu()
    if hasattr(sample_input, 'cpu'):
        sample_input = sample_input.cpu()
    if hasattr(sample_pl, 'cpu'):
        sample_pl = sample_pl.cpu()
    
    # Unnormalize target (ground truth)
    target_reshaped = sample_target.view(1, -1)  # Add batch dim
    _, unnorm_target = unnormalize_predictions(
        torch.zeros_like(target_reshaped), target_reshaped, mean, std, forecast_horizon
    )
    unnorm_target = unnorm_target[0]  # Remove batch dim: [forecast_horizon, features]
    
    # Unnormalize input (history)
    input_features = sample_input[:, :-1]  # Exclude p-level column
    unnorm_input = input_features * std[:input_features.shape[1]] + mean[:input_features.shape[1]]
    
    # Get p-levels from input and forecast - need to unnormalize them
    input_pl = sample_input[:, -1]  # p-level from input (normalized)
    forecast_pl = sample_pl  # p-level for forecast period (normalized)
    
    # Unnormalize p-levels (p-level is the last feature, index 11)
    pl_mean = mean[11]  # p-level mean
    pl_std = std[11]   # p-level std
    input_pl_unnorm = input_pl * pl_std + pl_mean
    forecast_pl_unnorm = forecast_pl * pl_std + pl_mean
    
    # Create time axis - connect predictions to last input step
    input_time = np.arange(0, len(input_pl_unnorm))
    forecast_time = np.arange(len(input_pl_unnorm), len(input_pl_unnorm) + forecast_horizon)
    
    # Get last input values for connection
    last_input_map = unnorm_input[-1, 0]  # Last MAP value from input
    last_input_pl = input_pl_unnorm[-1]   # Last P-level value from input
    
    # Set up subplots
    baseline_names = list(sample_predictions.keys())
    n_baselines = len(baseline_names)
    
    fig, axes = plt.subplots(1, n_baselines, figsize=(5*n_baselines, 5))
    if n_baselines == 1:
        axes = [axes]
    
    # Color scheme
    colors = {
        'history_map': 'tab:red',
        'ground_truth': 'tab:red', 
        'prediction': 'tab:red',
        'history_pl': 'tab:blue',
        'forecast_pl': 'tab:blue'
    }
    
    # Calculate global y-axis limits for consistent scaling
    all_map_values = []
    all_pl_values = []
    
    # Collect all MAP values
    all_map_values.extend(unnorm_input[:, 0].tolist())  # History MAP
    all_map_values.extend(unnorm_target[:, 0].tolist())  # Ground truth MAP
    
    # Collect all P-level values
    all_pl_values.extend(input_pl_unnorm.tolist())  # History P-levels
    all_pl_values.extend(forecast_pl_unnorm.tolist())  # Forecast P-levels
    
    # Process all MC samples to get uncertainty bounds for axis limits
    for baseline_name, predictions in sample_predictions.items():
        # predictions shape: [num_mc_samples, batch_size, features]
        mc_samples = predictions[:, 0, :]  # [num_mc_samples, features]
        
        # Unnormalize all MC samples
        for mc_idx in range(mc_samples.shape[0]):
            pred_reshaped = mc_samples[mc_idx].view(1, -1)
            unnorm_pred_temp, _ = unnormalize_predictions(
                pred_reshaped, torch.zeros_like(pred_reshaped), mean, std, forecast_horizon
            )
            all_map_values.extend(unnorm_pred_temp[0][:, 0].tolist())  # Predicted MAP
    
    # Calculate axis limits with some padding
    map_min, map_max = min(all_map_values), max(all_map_values)
    map_range = map_max - map_min
    map_y_limits = (map_min - 0.1 * map_range, map_max + 0.1 * map_range)
    
    pl_min, pl_max = 2.0, 10.0
    pl_y_limits = (pl_min, pl_max)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    for i, (baseline_name, predictions) in enumerate(sample_predictions.items()):
        ax1 = axes[i]
        
        # Process MC samples for uncertainty quantification
        # predictions shape: [num_mc_samples, batch_size, features]
        mc_samples = predictions[:, 0, :]  # [num_mc_samples, features]
        
        # Unnormalize all MC samples
        unnorm_mc_samples = []
        for mc_idx in range(mc_samples.shape[0]):
            pred_reshaped = mc_samples[mc_idx].view(1, -1)
            unnorm_pred, _ = unnormalize_predictions(
                pred_reshaped, torch.zeros_like(pred_reshaped), mean, std, forecast_horizon
            )
            unnorm_mc_samples.append(unnorm_pred[0])  # [forecast_horizon, features]
        
        # Stack to get shape: [num_mc_samples, forecast_horizon, features]
        unnorm_mc_samples = torch.stack(unnorm_mc_samples, dim=0)
        
        # Calculate mean and confidence intervals
        pred_mean = torch.mean(unnorm_mc_samples, dim=0)  # [forecast_horizon, features]
        pred_lower = torch.quantile(unnorm_mc_samples, lower_percentile/100.0, dim=0)  # [forecast_horizon, features]
        pred_upper = torch.quantile(unnorm_mc_samples, upper_percentile/100.0, dim=0)  # [forecast_horizon, features]
        
        # Set up time axis formatting
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/6:.1f}"))
        
        # Plot MAP on left y-axis
        ax1.set_xlabel("Time (hr)", size=20)
        ax1.tick_params(axis='x', labelsize=16)  # Increase the font size of x-axis tick labels
        if i ==0 :
            ax1.set_ylabel("MAP (mmHg)", size=20, color=colors['ground_truth'],  labelpad=1)
        ax1.tick_params(axis='y', labelsize=16)  # Increase the font size of y-axis tick labels
        
        # Plot history MAP
        ax1.plot(input_time, unnorm_input[:, 0], 
                color=colors['history_map'], linewidth=2, 
                alpha=0.5, linestyle='--', label='History MAP')
        
        # Plot ground truth MAP (connected to last input)
        ground_truth_time = np.concatenate([[input_time[-1]], forecast_time])
        ground_truth_map = np.concatenate([[last_input_map], unnorm_target[:, 0]])
        ax1.plot(ground_truth_time, ground_truth_map, 
                color=colors['ground_truth'], linewidth=2,
                alpha=0.5, linestyle='--', label='Ground Truth MAP')
        
        # Plot predicted MAP mean (connected to last input)
        predicted_time = np.concatenate([[input_time[-1]], forecast_time])
        predicted_map_mean = np.concatenate([[last_input_map], pred_mean[:, 0]])
        ax1.plot(predicted_time, predicted_map_mean, 
                color=colors['prediction'], linewidth=2, label=f'Mean Predicted MAP')
        
        # Plot uncertainty bands for MAP
        #predicted_map_lower = np.concatenate([[last_input_map], pred_lower[:, 0]])
        #predicted_map_upper = np.concatenate([[last_input_map], pred_upper[:, 0]])
        ax1.fill_between(forecast_time, pred_lower[:, 0], pred_upper[:, 0],
                        color=colors['prediction'], alpha=uncertainty_alpha, 
                        label=f'{int(confidence_level*100)}% Confidence')
        
        names = {
            'mlp_dropout': 'MLP',
            'neural_process': 'Neural Process',
            'clmu': 'CLMU',
            'state_space': 'State Space',
            'rotary_transformer': 'Transformer',
            'regular_transformer': 'Transformer'
        }
        ax1.tick_params(axis="y", labelcolor=colors['ground_truth'])
        ax1.set_title(f'{names.get(baseline_name, baseline_name)}', fontsize=20, fontweight='bold')
        
        # Set consistent y-axis limits for MAP
        ax1.set_ylim(map_y_limits)
        # Create second y-axis for p-level
        ax2 = ax1.twinx()
        if i == len(sample_predictions) - 1:
            ax2.set_ylabel("P-Level", size=20, color=colors['forecast_pl'], labelpad=1)
        ax2.tick_params(axis='y', labelsize=16)  # Increase the font size of y-axis tick labels

        # Plot p-levels (unnormalized, connected)
        ax2.plot(input_time, input_pl_unnorm, 
                color=colors['history_pl'], linewidth=2, linestyle='--', label='P-Level')
        
        # Plot forecast p-levels (connected to last input)
        forecast_pl_time = np.concatenate([[input_time[-1]], forecast_time])
        forecast_pl_connected = np.concatenate([[last_input_pl], forecast_pl_unnorm])
        ax2.plot(forecast_pl_time, forecast_pl_connected, 
                color=colors['forecast_pl'], linewidth=2, linestyle='--')#, label='Forecast P-Level')
        
        ax2.tick_params(axis="y", size=20, labelcolor=colors['forecast_pl'])
        
        # Set consistent y-axis limits for P-level
        ax2.set_ylim(pl_y_limits)
        
        # Add vertical line to separate history from forecast
        ax1.axvline(x=len(input_pl_unnorm), color='gray', linestyle=':', alpha=0.7)
        
        # Add legends
        if i == 0:  # Only add legend to first subplot
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
    
    labels = labels1 + labels2
    fig.legend(lines1 + lines2, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                fontsize=20, ncol=len(labels))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_baselines_from_results(results_dir: str, data_path: str, 
                                   baseline_names: List[str] = None,
                                   device: str = 'cuda:1', num_samples: int = 3,
                                   save_dir: Optional[str] = None):
    """Load baseline models from results directory and create comparison visualizations.
    
    Args:
        results_dir: Directory containing trained baseline models
        data_path: Path to the data file
        baseline_names: List of baseline names to visualize (default: all available)
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_dir: Directory to save plots (optional)
    """
    
    # Load and prepare data
    print("Loading and preparing data...")
    normalized_data = load_and_normalize_data(data_path, drop_col_11=True)
    train_dataset, val_dataset, test_dataset = create_datasets(
        normalized_data, input_horizon=6, output_horizon=6
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Data dimensions
    num_features = 12
    forecast_horizon = 6
    output_features = num_features - 1  # Exclude p-level
    output_dim = forecast_horizon * output_features
    mlp_input_dim = 6 * num_features + forecast_horizon  # flattened input + p-levels
    
    # Find available baseline model directories
    available_baselines = {}
    baseline_types = {
        'mlp': 'mlp_dropout',
        'neural_process': 'neural_process', 
        'clmu': 'clmu',
        'state_space': 'state_space',
        'rotary_transformer': 'rotary_transformer',
        'regular_transformer': 'regular_transformer'
    }
    
    # Define transformer model paths
    transformer_paths = {
        'rotary_transformer': [
            "/abiomed/downsampled/models/rotary_1hr_subsampled1.pth",
            "/abiomed/downsampled/models/rotary_1hr_subsampled2.pth",
            "/abiomed/downsampled/models/rotary_1hr_subsampled3.pth"
        ],
        'regular_transformer': [
            "/abiomed/downsampled/models/transformer_1hr_subsampled1.pth",
            "/abiomed/downsampled/models/transformer_1hr_subsampled2.pth",
            "/abiomed/downsampled/models/transformer_1hr_subsampled3.pth"
        ]
    }
    
    # Check baseline models in results directory
    if os.path.exists(results_dir):
        for folder in os.listdir(results_dir):
            folder_path = os.path.join(results_dir, folder)
            if os.path.isdir(folder_path):
                for key, baseline_type in baseline_types.items():
                    if key in folder.lower() and 'models' in folder.lower() and baseline_type not in ['rotary_transformer', 'regular_transformer']:
                        # Find a model file in this directory
                        model_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
                        if model_files:
                            model_path = os.path.join(folder_path, model_files[0])  # Use first model
                            available_baselines[baseline_type] = model_path
                            print(f"Found {baseline_type} model: {model_path}")
    
    # Check transformer models at predefined paths
    for transformer_type, paths in transformer_paths.items():
        for model_path in paths:
            if os.path.exists(model_path):
                available_baselines[transformer_type] = model_path
                print(f"Found {transformer_type} model: {model_path}")
                break  # Use first available model
    
    if not available_baselines:
        print(f"No baseline models found in {results_dir}")
        return
    
    # Filter by requested baselines
    if baseline_names:
        available_baselines = {k: v for k, v in available_baselines.items() 
                             if k in baseline_names}
    
    print(f"Loading {len(available_baselines)} baseline models...")
    
    # Load baseline models
    models_dict = {}
    for baseline_name, model_path in available_baselines.items():
        print(f"Loading {baseline_name} from {model_path}")
        
        if baseline_name == 'mlp_dropout':
            input_dim = mlp_input_dim
        else:
            input_dim = num_features
            
        model = load_baseline_model(model_path, baseline_name, device, input_dim, output_dim, normalized_data)
        models_dict[baseline_name] = model
    
    # Generate predictions
    print("Generating predictions...")
    predictions_data = generate_baseline_predictions(
        models_dict, test_loader, device, num_samples=num_samples
    )
    
    # Create visualizations
    print(f"Creating visualizations for {num_samples} samples...")
    for sample_idx in range(num_samples):
        print(f"Plotting sample {sample_idx + 1}/{num_samples}")
        
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"baseline_comparison_sample_{sample_idx + 1}.png")
        
        plot_baseline_comparison(
            predictions_data, 
            normalized_data['mean'], 
            normalized_data['std'],
            sample_idx=sample_idx,
            forecast_horizon=forecast_horizon,
            save_path=save_path,
            show_plot=(save_dir is None)  # Only show if not saving
        )
    
    print("Visualization complete!")

def find_changing_p_level_samples(data_loader, mean: torch.Tensor, std: torch.Tensor, 
                                 min_p_change: float = 1.0, max_samples: int = 50) -> List[int]:
    """Find samples where p-levels change significantly during the forecast period.
    
    Args:
        data_loader: DataLoader containing samples
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        min_p_change: Minimum change in p-level to consider (unnormalized units)
        max_samples: Maximum number of samples to search through
        
    Returns:
        List of sample indices with changing p-levels
    """
    
    pl_mean = mean[11].item()  # p-level mean
    pl_std = std[11].item()   # p-level std
    
    changing_samples = []
    
    with torch.no_grad():
        for i, (inputs, pl, targets) in enumerate(data_loader):
            if i >= max_samples:
                break
            
            # Get p-levels from input (history) and forecast
            input_pl = inputs[0, :, -1]  # p-level from input history (normalized)
            forecast_pl = pl[0]  # p-level for forecast period (normalized)
            
            # Unnormalize p-levels
            input_pl_unnorm = input_pl * pl_std + pl_mean
            forecast_pl_unnorm = forecast_pl * pl_std + pl_mean
            
            # Check if there's significant change
            last_input_pl = input_pl_unnorm[-1].item()
            forecast_pl_values = forecast_pl_unnorm.cpu().numpy()
            
            # Calculate maximum change from last input p-level
            max_change = max(abs(last_input_pl - pl_val) for pl_val in forecast_pl_values)
            
            # Also check for change within forecast period
            forecast_range = max(forecast_pl_values) - min(forecast_pl_values)
            
            if max_change >= min_p_change or forecast_range >= min_p_change:
                changing_samples.append(i)
    
    return changing_samples


def visualize_baselines_changing_p_levels(results_dir: str, data_path: str, 
                                        baseline_names: List[str] = None,
                                        device: str = 'cuda:1', min_p_change: float = 1.0,
                                        max_samples_search: int = 500, num_visualize: int = 5,
                                        save_dir: Optional[str] = None):
    """Load baseline models and create visualizations for samples with changing p-levels.
    
    Args:
        results_dir: Directory containing trained baseline models
        data_path: Path to the data file
        baseline_names: List of baseline names to visualize (default: all available)
        device: Device to run inference on
        min_p_change: Minimum p-level change to consider (unnormalized units)
        max_samples_search: Maximum samples to search through for changing p-levels
        num_visualize: Number of changing p-level samples to visualize
        save_dir: Directory to save plots (optional)
    """
    
    # Load and prepare data
    print("Loading and preparing data...")
    normalized_data = load_and_normalize_data(data_path, drop_col_11=True)
    train_dataset, val_dataset, test_dataset = create_datasets(
        normalized_data, input_horizon=6, output_horizon=6
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Find samples with changing p-levels
    print(f"Searching for samples with changing p-levels (min change: {min_p_change})...")
    changing_samples = find_changing_p_level_samples(
        test_loader, normalized_data['mean'], normalized_data['std'], 
        min_p_change, max_samples_search
    )
    
    print(f"Found {len(changing_samples)} samples with changing p-levels")
    if len(changing_samples) == 0:
        print("No samples found with significant p-level changes. Try reducing min_p_change.")
        return
    
    # Limit to requested number
    changing_samples = changing_samples[:num_visualize]
    print(f"Will visualize {len(changing_samples)} samples with changing p-levels")
    
    # Data dimensions
    num_features = 12
    forecast_horizon = 6
    output_features = num_features - 1  # Exclude p-level
    output_dim = forecast_horizon * output_features
    mlp_input_dim = 6 * num_features + forecast_horizon  # flattened input + p-levels
    
    # Find available baseline model directories (same as original function)
    available_baselines = {}
    baseline_types = {
        'mlp': 'mlp_dropout',
        'neural_process': 'neural_process', 
        'clmu': 'clmu',
        'state_space': 'state_space',
        'rotary_transformer': 'rotary_transformer',
        'regular_transformer': 'regular_transformer'
    }
    
    # Define transformer model paths
    transformer_paths = {
        'rotary_transformer': [
            "/abiomed/downsampled/models/rotary_1hr_subsampled1.pth",
            "/abiomed/downsampled/models/rotary_1hr_subsampled2.pth",
            "/abiomed/downsampled/models/rotary_1hr_subsampled3.pth"
        ],
        'regular_transformer': [
            "/abiomed/downsampled/models/transformer_1hr_subsampled1.pth",
            "/abiomed/downsampled/models/transformer_1hr_subsampled2.pth",
            "/abiomed/downsampled/models/transformer_1hr_subsampled3.pth"
        ]
    }
    
    # Check baseline models in results directory
    if os.path.exists(results_dir):
        for folder in os.listdir(results_dir):
            folder_path = os.path.join(results_dir, folder)
            if os.path.isdir(folder_path):
                for key, baseline_type in baseline_types.items():
                    if key in folder.lower() and 'models' in folder.lower() and baseline_type not in ['rotary_transformer', 'regular_transformer']:
                        model_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
                        if model_files:
                            model_path = os.path.join(folder_path, model_files[0])
                            available_baselines[baseline_type] = model_path
                            print(f"Found {baseline_type} model: {model_path}")
    
    # Check transformer models at predefined paths
    for transformer_type, paths in transformer_paths.items():
        for model_path in paths:
            if os.path.exists(model_path):
                available_baselines[transformer_type] = model_path
                print(f"Found {transformer_type} model: {model_path}")
                break
    
    if not available_baselines:
        print(f"No baseline models found in {results_dir}")
        return
    
    # Filter by requested baselines
    if baseline_names:
        available_baselines = {k: v for k, v in available_baselines.items() 
                             if k in baseline_names}
    
    print(f"Loading {len(available_baselines)} baseline models...")
    
    # Load baseline models
    models_dict = {}
    for baseline_name, model_path in available_baselines.items():
        print(f"Loading {baseline_name} from {model_path}")
        
        if baseline_name == 'mlp_dropout':
            input_dim = mlp_input_dim
        else:
            input_dim = num_features
            
        model = load_baseline_model(model_path, baseline_name, device, input_dim, output_dim, normalized_data)
        models_dict[baseline_name] = model
    
    # Generate predictions for changing p-level samples
    print("Generating predictions for changing p-level samples...")
    predictions_data = generate_baseline_predictions(
        models_dict, test_loader, device, sample_indices=changing_samples
    )
    
    # Create visualizations
    print(f"Creating visualizations for {len(changing_samples)} changing p-level samples...")
    for i, sample_idx in enumerate(changing_samples):
        print(f"Plotting changing p-level sample {i + 1}/{len(changing_samples)} (original index: {sample_idx})")
        
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"baseline_comparison_changing_pl_sample_{sample_idx}.png")
        
        plot_baseline_comparison(
            predictions_data, 
            normalized_data['mean'], 
            normalized_data['std'],
            sample_idx=i,  # Use position in filtered list
            forecast_horizon=forecast_horizon,
            save_path=save_path,
            show_plot=(save_dir is None)
        )
    
    print("Changing p-level visualization complete!")
    
    # Print summary of p-level changes found
    print("\nSummary of p-level changes in visualized samples:")
    pl_mean = normalized_data['mean'][11].item()
    pl_std = normalized_data['std'][11].item()
    
    for i, sample_idx in enumerate(changing_samples):
        sample_input = predictions_data['inputs'][i][0].cpu()  # [seq_len, features]
        sample_pl = predictions_data['pl'][i][0].cpu()  # [forecast_horizon]
        
        # Unnormalize p-levels
        input_pl_unnorm = sample_input[:, -1] * pl_std + pl_mean
        forecast_pl_unnorm = sample_pl * pl_std + pl_mean
        
        last_input_pl = input_pl_unnorm[-1].item()
        forecast_range = forecast_pl_unnorm.max().item() - forecast_pl_unnorm.min().item()
        max_change = max(abs(last_input_pl - pl_val) for pl_val in forecast_pl_unnorm)
        
        print(f"Sample {sample_idx}: Last input p-level: {last_input_pl:.1f}, "
              f"Forecast range: {forecast_range:.1f}, Max change: {max_change:.1f}")

def main():
    """Example usage of the visualization functions."""
    
    # Example parameters
    results_dir = "../results"  # Adjust path as needed
    data_path = "/abiomed/downsampled/10min_1hr_all_data.pkl"
    
    # Create standard visualizations
    visualize_baselines_from_results(
        results_dir=results_dir,
        data_path=data_path,
        baseline_names=['mlp_dropout', 'neural_process', 'rotary_transformer'],  # Specify which baselines to plot
        num_samples=3,
        save_dir="visualizations"  # Save plots here
    )
    
    # Create visualizations for samples with changing p-levels
    visualize_baselines_changing_p_levels(
        results_dir=results_dir,
        data_path=data_path,
        baseline_names=['mlp_dropout', 'neural_process', 'rotary_transformer'],
        min_p_change=1.0,  # Minimum p-level change to consider
        max_samples_search=500,  # Search through first 500 samples
        num_visualize=5,  # Visualize 5 samples with changing p-levels
        save_dir="visualizations_changing_pl"  # Save plots here
    )

if __name__ == "__main__":
    main()