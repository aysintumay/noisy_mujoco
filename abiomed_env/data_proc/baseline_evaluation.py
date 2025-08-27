"""
Main script for evaluating baseline models and comparing with TimeSeriesTransformer.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List

sys.path.append("..")
from data_utils import load_and_normalize_data, create_datasets
from baselines import MLPDropoutBaseline, NeuralProcessBaseline, CLMUBaseline, StateSpaceBaseline, BaselineTrainer
from evaluation_metrics import evaluate_model

def calculate_run_statistics(run_results: List[Dict]) -> Dict:
    """Calculate mean and std from multiple runs."""
    if not run_results:
        return {}
    
    # Get all metric names from first run
    metric_names = set()
    for result in run_results:
        metric_names.update(result.keys())
    
    statistics = {}
    for metric in metric_names:
        values = []
        for result in run_results:
            if metric in result:
                value = result[metric]
                if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                    values.append(float(value))
        
        if values:
            statistics[f"{metric}_mean"] = np.mean(values)
            statistics[f"{metric}_std"] = np.std(values)
            statistics[f"{metric}_runs"] = values
        else:
            statistics[f"{metric}_mean"] = np.nan
            statistics[f"{metric}_std"] = np.nan
            statistics[f"{metric}_runs"] = []
    
    return statistics

def load_pretrained_baseline_models(models_dir: str, model_type: str, device: str,
                                   input_dim: int, output_dim: int, 
                                   test_loader, mean, std, raw_test_data) -> List[Dict]:
    """Load and evaluate pretrained baseline models.
    
    Args:
        models_dir: Directory containing the pretrained models
        model_type: 'mlp' or 'neural_process'
        device: Device to run on
        input_dim: Input dimension for model
        output_dim: Output dimension for model
        test_loader: Test data loader
        mean: Feature means for unnormalization
        std: Feature stds for unnormalization
        raw_test_data: Raw test data for static/dynamic split
    
    Returns:
        List of evaluation results from each model
    """
    results = []
    
    if not os.path.exists(models_dir):
        print(f"Pretrained models directory not found: {models_dir}")
        return results
    
    # Find all model files in the directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    model_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(model_files)} pretrained {model_type} models in {models_dir}")
    
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(models_dir, model_file)
        print(f"Loading pretrained {model_type} model {i+1}/{len(model_files)}: {model_file}")
        
        try:
            # Create model instance
            if model_type == 'mlp':
                model = MLPDropoutBaseline(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dims=[512, 256, 128],
                    dropout_rate=0.2,
                    device=device
                )
            elif model_type == 'neural_process':
                model = NeuralProcessBaseline(
                    input_dim=12,  # num_features
                    output_dim=output_dim,
                    latent_dim=128,
                    hidden_dim=256,
                    device=device
                )
            elif model_type == 'clmu':
                model = CLMUBaseline(
                    input_dim=12,  # num_features
                    output_dim=output_dim,
                    memory_dim=64,
                    hidden_dim=128,
                    num_layers=2,
                    device=device
                )
            elif model_type == 'state_space':
                model = StateSpaceBaseline(
                    input_dim=12,  # num_features
                    output_dim=output_dim,
                    state_dim=64,
                    hidden_dim=128,
                    device=device
                )
            else:
                print(f"Unknown model type: {model_type}")
                continue
            
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Evaluate model
            print(f"Evaluating pretrained {model_type} model: {model_file}")
            metrics = evaluate_model(
                model,
                test_loader,
                mean,
                std,
                raw_test_data,
                forecast_horizon=6,
                num_samples=50,
                device=device
            )
            
            results.append(metrics)
            
            print(f"Pretrained {model_type} model ({model_file}) Results:")
            key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']
            for metric in key_metrics:
                value = metrics.get(metric, np.nan)
                if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: NaN")
                    
        except Exception as e:
            print(f"Error loading/evaluating model {model_file}: {e}")
            continue
    
    return results

def evaluate_baselines(data_path: str, results_dir: str = "../results", 
                      device: str = 'cuda:1', num_runs: int = 3, retrain: bool = True,
                      pretrained_models_dir: str = None, baselines_to_run: list = None) -> Dict:
    """Evaluate selected baseline models.
    
    Args:
        data_path: Path to the data file
        results_dir: Directory to save results
        device: Device to run on
        num_runs: Number of runs for statistical analysis
        retrain: If True, retrain models. If False, load from pretrained_models_dir
        pretrained_models_dir: Directory containing pretrained models (when retrain=False)
        baselines_to_run: List of baselines to run ['mlp_dropout', 'neural_process', 'clmu', 'state_space', 'transformers'] or ['all']
    """
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Handle baseline selection
    if baselines_to_run is None:
        baselines_to_run = ['all']
    
    if 'all' in baselines_to_run:
        baselines_to_run = ['mlp_dropout', 'neural_process', 'clmu', 'state_space', 'transformers']
    
    print(f"Baselines to run: {baselines_to_run}")
    print("Loading and preparing data...")
    # Load data
    normalized_data = load_and_normalize_data(data_path, drop_col_11=True)
    train_dataset, val_dataset, test_dataset = create_datasets(
        normalized_data, input_horizon=6, output_horizon=6
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Data dimensions
    num_features = 12  # After dropping column 11
    forecast_horizon = 6
    output_features = num_features - 1  # Exclude p-level
    output_dim = forecast_horizon * output_features
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Evaluate MLP + Dropout Baseline
    if 'mlp_dropout' in baselines_to_run:
        print("\n" + "="*60)
        if retrain:
            print(f"TRAINING MLP + DROPOUT BASELINE ({num_runs} runs)")
        else:
            print("LOADING PRETRAINED MLP + DROPOUT BASELINE")
        print("="*60)
        
        mlp_input_dim = 6 * num_features + forecast_horizon  # flattened input + p-levels
        mlp_run_results = []
        
        if retrain:
            # Train new models
            mlp_models_dir = os.path.join(results_dir, f"mlp_models_{timestamp}")
            os.makedirs(mlp_models_dir, exist_ok=True)
            
            for run_idx in range(num_runs):
                print(f"\n--- MLP Run {run_idx + 1}/{num_runs} ---")
                
                # Create new model for each run
                mlp_model = MLPDropoutBaseline(
                    input_dim=mlp_input_dim,
                    output_dim=output_dim,
                    hidden_dims=[512, 256, 128],
                    dropout_rate=0.2,
                    device=device
                )
                
                # Train MLP
                mlp_trainer = BaselineTrainer(mlp_model, device=device)
                print(f"Training MLP baseline run {run_idx + 1}...")
                mlp_training_results = mlp_trainer.train_model(
                    train_loader, val_loader, num_epochs=50, learning_rate=0.001
                )
                
                print(f"MLP run {run_idx + 1} training completed. Best val loss: {mlp_training_results['best_val_loss']:.4f}")
                
                # Evaluate MLP
                print(f"Evaluating MLP baseline run {run_idx + 1}...")
                mlp_metrics = evaluate_model(
                    mlp_model,
                    test_loader,
                    normalized_data['mean'],
                    normalized_data['std'],
                    normalized_data['raw_test'],
                    forecast_horizon=6,
                    num_samples=50,
                    device=device
                )
                
                mlp_metrics.update(mlp_training_results)
                mlp_run_results.append(mlp_metrics)
                
                # Save model
                mlp_model_path = os.path.join(mlp_models_dir, f"mlp_baseline_run_{run_idx + 1}.pth")
                torch.save(mlp_model.state_dict(), mlp_model_path)
                print(f"MLP run {run_idx + 1} model saved to {mlp_model_path}")
                
                # Print individual run results
                print(f"MLP Run {run_idx + 1} Results:")
                for metric, value in mlp_metrics.items():
                    if 'loss' in metric or metric in ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']:
                        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
        else:
            # Load pretrained models
            if pretrained_models_dir:
                mlp_pretrained_dir = os.path.join(pretrained_models_dir, "mlp_models")
                if not os.path.exists(mlp_pretrained_dir):
                    # Try alternative naming patterns
                    mlp_dirs = [d for d in os.listdir(pretrained_models_dir) if 'mlp' in d.lower()]
                    if mlp_dirs:
                        mlp_pretrained_dir = os.path.join(pretrained_models_dir, mlp_dirs[0])
                        print(f"Using MLP models from: {mlp_pretrained_dir}")
            else:
                print("Error: pretrained_models_dir must be provided when retrain=False")
                mlp_pretrained_dir = None
            
            if mlp_pretrained_dir and os.path.exists(mlp_pretrained_dir):
                mlp_run_results = load_pretrained_baseline_models(
                    mlp_pretrained_dir, 'mlp', device, mlp_input_dim, output_dim,
                    test_loader, normalized_data['mean'], normalized_data['std'], normalized_data['raw_test']
                )
                mlp_models_dir = mlp_pretrained_dir
            else:
                print("Could not find pretrained MLP models. Skipping MLP evaluation.")
                mlp_run_results = []
                mlp_models_dir = None
        
        # Calculate statistics across runs
        if mlp_run_results:
            mlp_statistics = calculate_run_statistics(mlp_run_results)
            results['mlp_dropout'] = {
                'individual_runs': mlp_run_results,
                'statistics': mlp_statistics,
                'models_dir': mlp_models_dir
            }
            
            print(f"\n=== MLP FINAL RESULTS (Mean ± Std over {len(mlp_run_results)} runs) ===")
            key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps', 'best_val_loss']
            for metric in key_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in mlp_statistics:
                    mean_val = mlp_statistics[mean_key]
                    std_val = mlp_statistics[std_key]
                    if not np.isnan(mean_val):
                        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"  {metric}: NaN")
    else:
        print("Skipping MLP + Dropout Baseline (not in selected baselines)")
    
    # 2. Evaluate Neural Process Baseline
    if 'neural_process' in baselines_to_run:
        print("\n" + "="*60)
        if retrain:
            print(f"TRAINING NEURAL PROCESS BASELINE ({num_runs} runs)")
        else:
            print("LOADING PRETRAINED NEURAL PROCESS BASELINE")
        print("="*60)
        
        np_run_results = []
        
        if retrain:
            # Train new models
            np_models_dir = os.path.join(results_dir, f"neural_process_models_{timestamp}")
            os.makedirs(np_models_dir, exist_ok=True)
            
            for run_idx in range(num_runs):
                print(f"\n--- Neural Process Run {run_idx + 1}/{num_runs} ---")
                
                # Create new model for each run
                np_model = NeuralProcessBaseline(
                    input_dim=num_features,
                    output_dim=output_dim,
                    latent_dim=128,
                    hidden_dim=256,
                    device=device
                )
                
                # Train Neural Process
                np_trainer = BaselineTrainer(np_model, device=device)
                print(f"Training Neural Process baseline run {run_idx + 1}...")
                np_training_results = np_trainer.train_model(
                    train_loader, val_loader, num_epochs=50, learning_rate=0.001
                )
                
                print(f"Neural Process run {run_idx + 1} training completed. Best val loss: {np_training_results['best_val_loss']:.4f}")
                
                # Evaluate Neural Process
                print(f"Evaluating Neural Process baseline run {run_idx + 1}...")
                np_metrics = evaluate_model(
                    np_model,
                    test_loader,
                    normalized_data['mean'],
                    normalized_data['std'],
                    normalized_data['raw_test'],
                    forecast_horizon=6,
                    num_samples=50,
                    device=device
                )
                
                np_metrics.update(np_training_results)
                np_run_results.append(np_metrics)
                
                # Save model
                np_model_path = os.path.join(np_models_dir, f"neural_process_baseline_run_{run_idx + 1}.pth")
                torch.save(np_model.state_dict(), np_model_path)
                print(f"Neural Process run {run_idx + 1} model saved to {np_model_path}")
                
                # Print individual run results
                print(f"Neural Process Run {run_idx + 1} Results:")
                for metric, value in np_metrics.items():
                    if 'loss' in metric or metric in ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']:
                        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
        else:
            # Load pretrained models  
            if pretrained_models_dir:
                np_pretrained_dir = os.path.join(pretrained_models_dir, "neural_process_models")
                if not os.path.exists(np_pretrained_dir):
                    # Try alternative naming patterns
                    np_dirs = [d for d in os.listdir(pretrained_models_dir) if 'neural_process' in d.lower()]
                    if np_dirs:
                        np_pretrained_dir = os.path.join(pretrained_models_dir, np_dirs[0])
                        print(f"Using Neural Process models from: {np_pretrained_dir}")
            else:
                print("Error: pretrained_models_dir must be provided when retrain=False")
                np_pretrained_dir = None
                
            if np_pretrained_dir and os.path.exists(np_pretrained_dir):
                np_run_results = load_pretrained_baseline_models(
                    np_pretrained_dir, 'neural_process', device, num_features, output_dim,
                    test_loader, normalized_data['mean'], normalized_data['std'], normalized_data['raw_test']
                )
                np_models_dir = np_pretrained_dir
            else:
                print("Could not find pretrained Neural Process models. Skipping Neural Process evaluation.")
                np_run_results = []
                np_models_dir = None
        
        # Calculate statistics across runs
        if np_run_results:
            np_statistics = calculate_run_statistics(np_run_results)
            results['neural_process'] = {
                'individual_runs': np_run_results,
                'statistics': np_statistics,
                'models_dir': np_models_dir
            }
            
            print(f"\n=== NEURAL PROCESS FINAL RESULTS (Mean ± Std over {len(np_run_results)} runs) ===")
            key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps', 'best_val_loss']
            for metric in key_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in np_statistics:
                    mean_val = np_statistics[mean_key]
                    std_val = np_statistics[std_key]
                    if not np.isnan(mean_val):
                        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"  {metric}: NaN")
    else:
        print("Skipping Neural Process Baseline (not in selected baselines)")
    # 3. Evaluate CLMU Baseline
    if 'clmu' in baselines_to_run:
        print("\n" + "="*60)
        if retrain:
            print(f"TRAINING CLMU BASELINE ({num_runs} runs)")
        else:
            print("LOADING PRETRAINED CLMU BASELINE")
        print("="*60)
        
        clmu_run_results = []
        
        if retrain:
            # Train new models
            clmu_models_dir = os.path.join(results_dir, f"clmu_models_{timestamp}")
            os.makedirs(clmu_models_dir, exist_ok=True)
            
            for run_idx in range(num_runs):
                print(f"\n--- CLMU Run {run_idx + 1}/{num_runs} ---")
                
                # Create new model for each run
                clmu_model = CLMUBaseline(
                    input_dim=num_features,
                    output_dim=output_dim,
                    memory_dim=64,
                    hidden_dim=128,
                    num_layers=2,
                    device=device
                )
                
                # Train CLMU
                clmu_trainer = BaselineTrainer(clmu_model, device=device)
                print(f"Training CLMU baseline run {run_idx + 1}...")
                clmu_training_results = clmu_trainer.train_model(
                    train_loader, val_loader, num_epochs=50, learning_rate=0.001
                )
                
                print(f"CLMU run {run_idx + 1} training completed. Best val loss: {clmu_training_results['best_val_loss']:.4f}")
                
                # Evaluate CLMU
                print(f"Evaluating CLMU baseline run {run_idx + 1}...")
                clmu_metrics = evaluate_model(
                    clmu_model,
                    test_loader,
                    normalized_data['mean'],
                    normalized_data['std'],
                    normalized_data['raw_test'],
                    forecast_horizon=6,
                    num_samples=50,
                    device=device
                )
                
                clmu_metrics.update(clmu_training_results)
                clmu_run_results.append(clmu_metrics)
                
                # Save model
                clmu_model_path = os.path.join(clmu_models_dir, f"clmu_baseline_run_{run_idx + 1}.pth")
                torch.save(clmu_model.state_dict(), clmu_model_path)
                print(f"CLMU run {run_idx + 1} model saved to {clmu_model_path}")
                
                # Print individual run results
                print(f"CLMU Run {run_idx + 1} Results:")
                for metric, value in clmu_metrics.items():
                    if 'loss' in metric or metric in ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']:
                        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
        else:
            # Load pretrained models  
            if pretrained_models_dir:
                clmu_pretrained_dir = os.path.join(pretrained_models_dir, "clmu_models")
                if not os.path.exists(clmu_pretrained_dir):
                    # Try alternative naming patterns
                    clmu_dirs = [d for d in os.listdir(pretrained_models_dir) if 'clmu' in d.lower()]
                    if clmu_dirs:
                        clmu_pretrained_dir = os.path.join(pretrained_models_dir, clmu_dirs[0])
                        print(f"Using CLMU models from: {clmu_pretrained_dir}")
            else:
                print("Error: pretrained_models_dir must be provided when retrain=False")
                clmu_pretrained_dir = None
                
            if clmu_pretrained_dir and os.path.exists(clmu_pretrained_dir):
                clmu_run_results = load_pretrained_baseline_models(
                    clmu_pretrained_dir, 'clmu', device, num_features, output_dim,
                    test_loader, normalized_data['mean'], normalized_data['std'], normalized_data['raw_test']
                )
                clmu_models_dir = clmu_pretrained_dir
            else:
                print("Could not find pretrained CLMU models. Skipping CLMU evaluation.")
                clmu_run_results = []
                clmu_models_dir = None
        
        # Calculate statistics across runs
        if clmu_run_results:
            clmu_statistics = calculate_run_statistics(clmu_run_results)
            results['clmu'] = {
                'individual_runs': clmu_run_results,
                'statistics': clmu_statistics,
                'models_dir': clmu_models_dir
            }
            
            print(f"\n=== CLMU FINAL RESULTS (Mean ± Std over {len(clmu_run_results)} runs) ===")
            key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps', 'best_val_loss']
            for metric in key_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in clmu_statistics:
                    mean_val = clmu_statistics[mean_key]
                    std_val = clmu_statistics[std_key]
                    if not np.isnan(mean_val):
                        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"  {metric}: NaN")
    else:
        print("Skipping CLMU Baseline (not in selected baselines)")
    # 4. Evaluate State-Space Model Baseline
    if 'state_space' in baselines_to_run:
        print("\n" + "="*60)
        if retrain:
            print(f"TRAINING STATE-SPACE MODEL BASELINE ({num_runs} runs)")
        else:
            print("LOADING PRETRAINED STATE-SPACE MODEL BASELINE")
        print("="*60)
        
        ssm_run_results = []
        
        if retrain:
            # Train new models
            ssm_models_dir = os.path.join(results_dir, f"state_space_models_{timestamp}")
            os.makedirs(ssm_models_dir, exist_ok=True)
            
            for run_idx in range(num_runs):
                print(f"\n--- State-Space Model Run {run_idx + 1}/{num_runs} ---")
                
                # Create new model for each run
                ssm_model = StateSpaceBaseline(
                    input_dim=num_features,
                    output_dim=output_dim,
                    state_dim=64,
                    hidden_dim=128,
                    device=device
                )
                
                # Train State-Space Model
                ssm_trainer = BaselineTrainer(ssm_model, device=device)
                print(f"Training State-Space Model baseline run {run_idx + 1}...")
                ssm_training_results = ssm_trainer.train_model(
                    train_loader, val_loader, num_epochs=50, learning_rate=0.001
                )
                
                print(f"State-Space Model run {run_idx + 1} training completed. Best val loss: {ssm_training_results['best_val_loss']:.4f}")
                
                # Evaluate State-Space Model
                print(f"Evaluating State-Space Model baseline run {run_idx + 1}...")
                ssm_metrics = evaluate_model(
                    ssm_model,
                    test_loader,
                    normalized_data['mean'],
                    normalized_data['std'],
                    normalized_data['raw_test'],
                    forecast_horizon=6,
                    num_samples=50,
                    device=device
                )
                
                ssm_metrics.update(ssm_training_results)
                ssm_run_results.append(ssm_metrics)
                
                # Save model
                ssm_model_path = os.path.join(ssm_models_dir, f"state_space_baseline_run_{run_idx + 1}.pth")
                torch.save(ssm_model.state_dict(), ssm_model_path)
                print(f"State-Space Model run {run_idx + 1} model saved to {ssm_model_path}")
                
                # Print individual run results
                print(f"State-Space Model Run {run_idx + 1} Results:")
                for metric, value in ssm_metrics.items():
                    if 'loss' in metric or metric in ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']:
                        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
        else:
            # Load pretrained models  
            if pretrained_models_dir:
                ssm_pretrained_dir = os.path.join(pretrained_models_dir, "state_space_models")
                if not os.path.exists(ssm_pretrained_dir):
                    # Try alternative naming patterns
                    ssm_dirs = [d for d in os.listdir(pretrained_models_dir) if 'state_space' in d.lower()]
                    if ssm_dirs:
                        ssm_pretrained_dir = os.path.join(pretrained_models_dir, ssm_dirs[0])
                        print(f"Using State-Space models from: {ssm_pretrained_dir}")
            else:
                print("Error: pretrained_models_dir must be provided when retrain=False")
                ssm_pretrained_dir = None
                
            if ssm_pretrained_dir and os.path.exists(ssm_pretrained_dir):
                ssm_run_results = load_pretrained_baseline_models(
                    ssm_pretrained_dir, 'state_space', device, num_features, output_dim,
                    test_loader, normalized_data['mean'], normalized_data['std'], normalized_data['raw_test']
                )
                ssm_models_dir = ssm_pretrained_dir
            else:
                print("Could not find pretrained State-Space models. Skipping State-Space evaluation.")
                ssm_run_results = []
                ssm_models_dir = None
        
        # Calculate statistics across runs
        if ssm_run_results:
            ssm_statistics = calculate_run_statistics(ssm_run_results)
            results['state_space'] = {
                'individual_runs': ssm_run_results,
                'statistics': ssm_statistics,
                'models_dir': ssm_models_dir
            }
            
            print(f"\n=== STATE-SPACE MODEL FINAL RESULTS (Mean ± Std over {len(ssm_run_results)} runs) ===")
            key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps', 'best_val_loss']
            for metric in key_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in ssm_statistics:
                    mean_val = ssm_statistics[mean_key]
                    std_val = ssm_statistics[std_key]
                    if not np.isnan(mean_val):
                        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"  {metric}: NaN")
    else:
        print("Skipping State-Space Baseline (not in selected baselines)")
    # 5. Evaluate existing TimeSeriesTransformer models
    if 'transformers' in baselines_to_run:
        print("\n" + "="*60)
        print("EVALUATING EXISTING TRANSFORMER MODELS")
        print("="*60)
        
        try:
            from model import WorldModel
            
            # Define transformer model groups
            # rotary_transformer_paths = [
            #     "/abiomed/downsampled/models/rotary_1hr_mae.pth",
            #     "/abiomed/downsampled/models/rotary_1hr_mse2.pth", 
            #     "/abiomed/downsampled/models/rotary_1hr_mse3.pth"
            # ]
            
            # regular_transformer_paths = [
            #     "/abiomed/downsampled/models/transformer_1hr_mse1.pth",
            #     "/abiomed/downsampled/models/transformer_1hr_mse2.pth",
            #     "/abiomed/downsampled/models/transformer_1hr_mse3.pth"
            # ]
            
            rotary_transformer_paths = [
                "/abiomed/downsampled/models/rotary_1hr_subsampled1.pth",
                "/abiomed/downsampled/models/rotary_1hr_subsampled2.pth",
                "/abiomed/downsampled/models/rotary_1hr_subsampled3.pth"
            ]
            
            regular_transformer_paths = [
                "/abiomed/downsampled/models/transformer_1hr_subsampled1.pth",
                "/abiomed/downsampled/models/transformer_1hr_subsampled2.pth",
                "/abiomed/downsampled/models/transformer_1hr_subsampled3.pth"
            ]

            # Evaluate Rotary Transformers
            print(f"\n--- Evaluating Rotary Transformers ---")
            rotary_run_results = []
            rotary_models_info = []
            
            for model_path in rotary_transformer_paths:
                if os.path.exists(model_path):
                    print(f"Evaluating rotary transformer: {os.path.basename(model_path)}")
                    
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
                    transformer.mean = normalized_data['mean']
                    transformer.std = normalized_data['std']
                    
                    # Evaluate
                    transformer_metrics = evaluate_model(
                        transformer.model,
                        test_loader,
                        normalized_data['mean'],
                        normalized_data['std'],
                        normalized_data['raw_test'],
                        forecast_horizon=6,
                        num_samples=50,
                        device=device
                    )
                    
                    rotary_run_results.append(transformer_metrics)
                    rotary_models_info.append(os.path.basename(model_path))
                    
                    print(f"Rotary transformer ({os.path.basename(model_path)}) Results:")
                    key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']
                    for metric in key_metrics:
                        value = transformer_metrics.get(metric, np.nan)
                        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
                else:
                    print(f"Model not found: {model_path}")
            
            # Calculate rotary transformer statistics
            if rotary_run_results:
                rotary_statistics = calculate_run_statistics(rotary_run_results)
                results['rotary_transformer'] = {
                    'individual_runs': rotary_run_results,
                    'statistics': rotary_statistics,
                    'models_evaluated': rotary_models_info
                }
                
                print(f"\n=== ROTARY TRANSFORMER FINAL RESULTS (Mean ± Std over {len(rotary_run_results)} models) ===")
                key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']
                for metric in key_metrics:
                    mean_key = f"{metric}_mean"
                    std_key = f"{metric}_std"
                    if mean_key in rotary_statistics:
                        mean_val = rotary_statistics[mean_key]
                        std_val = rotary_statistics[std_key]
                        if not np.isnan(mean_val):
                            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                        else:
                            print(f"  {metric}: NaN")
            
            # Evaluate Regular Transformers
            print(f"\n--- Evaluating Regular Transformers ---")
            regular_run_results = []
            regular_models_info = []
            
            for model_path in regular_transformer_paths:
                if os.path.exists(model_path):
                    print(f"Evaluating regular transformer: {os.path.basename(model_path)}")
                    
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
                    transformer.mean = normalized_data['mean']
                    transformer.std = normalized_data['std']
                    
                    # Evaluate
                    transformer_metrics = evaluate_model(
                        transformer.model,
                        test_loader,
                        normalized_data['mean'],
                        normalized_data['std'],
                        normalized_data['raw_test'],
                        forecast_horizon=6,
                        num_samples=50,
                        device=device
                    )
                    
                    regular_run_results.append(transformer_metrics)
                    regular_models_info.append(os.path.basename(model_path))
                    
                    print(f"Regular transformer ({os.path.basename(model_path)}) Results:")
                    key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']
                    for metric in key_metrics:
                        value = transformer_metrics.get(metric, np.nan)
                        if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
                else:
                    print(f"Model not found: {model_path}")
            
            # Calculate regular transformer statistics
            if regular_run_results:
                regular_statistics = calculate_run_statistics(regular_run_results)
                results['regular_transformer'] = {
                    'individual_runs': regular_run_results,
                    'statistics': regular_statistics,
                    'models_evaluated': regular_models_info
                }
                
                print(f"\n=== REGULAR TRANSFORMER FINAL RESULTS (Mean ± Std over {len(regular_run_results)} models) ===")
                key_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']
                for metric in key_metrics:
                    mean_key = f"{metric}_mean"
                    std_key = f"{metric}_std"
                    if mean_key in regular_statistics:
                        mean_val = regular_statistics[mean_key]
                        std_val = regular_statistics[std_key]
                        if not np.isnan(mean_val):
                            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
                        else:
                            print(f"  {metric}: NaN")
            
        except Exception as e:
            print(f"Could not evaluate transformer models: {e}")
            results['rotary_transformer'] = {'individual_runs': [], 'statistics': {}, 'models_evaluated': []}
            results['regular_transformer'] = {'individual_runs': [], 'statistics': {}, 'models_evaluated': []}
    else:
        print("Skipping Transformer Baselines (not in selected baselines)")
    # 6. Summary comparison
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON (Mean ± Std)")
    print("="*60)
    
    # Define all models to compare and all metrics
    models_to_compare = ['mlp_dropout', 'neural_process', 'clmu', 'state_space', 'rotary_transformer', 'regular_transformer']
    all_metrics = ['mae_all_features', 'mae_map_only', 'mae_static', 'mae_dynamic', 'trend_accuracy', 'crps']
    
    # Print header
    header = f"{'Model':<20}"
    for metric in all_metrics:
        if metric == 'mae_all_features':
            header += f"{'MAE All':<15}"
        elif metric == 'mae_map_only':
            header += f"{'MAE MAP':<15}"
        elif metric == 'mae_static':
            header += f"{'MAE Static':<15}"
        elif metric == 'mae_dynamic':
            header += f"{'MAE Dynamic':<15}"
        elif metric == 'trend_accuracy':
            header += f"{'Trend Acc':<15}"
        elif metric == 'crps':
            header += f"{'CRPS':<15}"
    
    print(header)
    print("-" * (20 + 15 * len(all_metrics)))
    
    # Print results for each model
    for model_name in models_to_compare:
        if model_name in results and 'statistics' in results[model_name]:
            statistics = results[model_name]['statistics']
            
            row = f"{model_name:<20}"
            
            for metric in all_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                
                mean_val = statistics.get(mean_key, np.nan)
                std_val = statistics.get(std_key, np.nan)
                
                if isinstance(mean_val, (int, float, np.floating, np.integer)) and not np.isnan(mean_val):
                    if isinstance(std_val, (int, float, np.floating, np.integer)) and not np.isnan(std_val):
                        metric_str = f"{mean_val:.2f}±{std_val:.2f}"
                    else:
                        metric_str = f"{mean_val:.2f}±0.00"
                else:
                    metric_str = "NaN"
                
                row += f"{metric_str:<15}"
            
            print(row)
        elif model_name in results:
            # Handle case where model exists but doesn't have statistics structure
            print(f"{model_name:<20} {'(evaluation failed)':<15}")
    
    print("\nLegend:")
    print("- MAE All: Mean Absolute Error across all features")
    print("- MAE MAP: Mean Absolute Error for MAP (Mean Arterial Pressure) only")
    print("- MAE Static: MAE for samples with static p-levels")
    print("- MAE Dynamic: MAE for samples with dynamic p-levels")  
    print("- Trend Acc: Trend direction accuracy for MAP")
    print("- CRPS: Continuous Ranked Probability Score (uncertainty quantification)")
    print("\nNote: All values shown as Mean ± Standard Deviation across multiple runs/models")
    
    # 5. Save all results
    results_filename = f"baseline_evaluation_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    
    # Convert to JSON-serializable format
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
            return None
        else:
            return obj
    
    json_results = convert_to_json_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Also save as pickle
    pickle_filename = f"baseline_evaluation_{timestamp}.pkl"
    pickle_path = os.path.join(results_dir, pickle_filename)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {results_path}")
    print(f"  Pickle: {pickle_path}")
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--retrain', action='store_true', 
                        help='Retrain models from scratch (default: False, load pretrained)')
    parser.add_argument('--pretrained-dir', type=str, 
                        default='../results/neural_process_models_20250816_080958',
                        help='Directory containing pretrained models (parent dir containing mlp_models/ and neural_process_models/)')
    parser.add_argument('--data-path', type=str, 
                        default='/abiomed/downsampled/10min_1hr_all_data_subsampled.pkl',
                        help='Path to the data file')
    parser.add_argument('--results-dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (default: auto-detect)')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs for statistical analysis (only used if --retrain)')
    parser.add_argument('--baselines', nargs='+', 
                        choices=['mlp_dropout', 'neural_process', 'clmu', 'state_space', 'transformers', 'all'],
                        default=['all'],
                        help='Which baselines to train/evaluate. Options: mlp_dropout, neural_process, clmu, state_space, transformers, all')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Running baseline evaluation on {device}")
    print(f"Data path: {args.data_path}")
    print(f"Results directory: {args.results_dir}")
    
    if args.retrain:
        print(f"Mode: RETRAINING models ({args.num_runs} runs each)")
        print(f"Selected baselines: {args.baselines}")
        results = evaluate_baselines(
            args.data_path, args.results_dir, device, args.num_runs, 
            retrain=True, pretrained_models_dir=None, baselines_to_run=args.baselines
        )
    else:
        print(f"Mode: LOADING pretrained models from {args.pretrained_dir}")
        print(f"Selected baselines: {args.baselines}")
        print(f"Number of models loaded will depend on what's available in the directories")
        
        # Extract the parent directory from the provided path
        if 'neural_process_models' in args.pretrained_dir:
            pretrained_parent = os.path.dirname(args.pretrained_dir)
        elif 'mlp_models' in args.pretrained_dir:
            pretrained_parent = os.path.dirname(args.pretrained_dir)
        else:
            pretrained_parent = args.pretrained_dir
            
        results = evaluate_baselines(
            args.data_path, args.results_dir, device, num_runs=1,  # num_runs ignored when loading
            retrain=False, pretrained_models_dir=pretrained_parent, baselines_to_run=args.baselines
        )
    
    print("\nBaseline evaluation completed!")
    
    return results

if __name__ == "__main__":
    main()