"""
Hyperparameter search for TimeSeriesTransformer model with Weights & Biases logging.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
import random
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

sys.path.append("..")
from model import WorldModel
from data_utils import load_and_normalize_data, create_datasets
from evaluation_metrics import evaluate_model

class HyperparameterSearch:
    """Hyperparameter search for TimeSeriesTransformer with wandb logging."""
    
    def __init__(self, data_path: str, results_dir: str = "../results", 
                 device: str = 'cuda:1', max_runs: int = 100, 
                 wandb_project: str = "timeseries_transformer_search",
                 wandb_entity: str = None, models_dir: str = "/abiomed/hp_search"):
        self.data_path = data_path
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.device = torch.device(device)
        self.max_runs = max_runs
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # Create results and models directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        print(f"Models will be saved to: {models_dir}")
        
        # Initialize wandb project (but don't start a run yet)
        if WANDB_AVAILABLE:
            print(f"Will log to wandb project: {self.wandb_project}")
        else:
            print("Running without wandb logging (wandb not available)")
        
        # Load and prepare data
        print("Loading data...")
        self.normalized_data = load_and_normalize_data(data_path, drop_col_11=True)
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(
            self.normalized_data, input_horizon=6, output_horizon=6
        )
        
        print(f"Dataset sizes - Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def get_hyperparameter_space(self) -> Dict:
        """Define hyperparameter search space."""
        return {
            'dim_model': [128, 256, 512],
            'num_heads': [4, 8, 16],
            'num_encoder_layers': [2, 3, 4],
            'num_decoder_layers': [1, 2, 3],
            'encoder_dropout': [0.0, 0.1, 0.2],
            'decoder_dropout': [0.0, 0.1, 0.2],
            'model_type': ['transformer', 'rotary_transformer'],
            'batch_size': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'num_epochs': [35]
        }
    
    def sample_hyperparameters(self, search_space: Dict) -> Dict:
        """Sample a random hyperparameter configuration."""
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        
        # Ensure head dimension compatibility
        while config['dim_model'] % config['num_heads'] != 0:
            config['num_heads'] = random.choice(search_space['num_heads'])
        
        return config
    
    def train_and_evaluate(self, config: Dict, run_id: int) -> Dict:
        """Train model with given config and evaluate."""
        print(f"\nRun {run_id}: Training with config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Initialize wandb run for this configuration
        wandb_run = None
        if WANDB_AVAILABLE:
            run_name = f"run_{run_id:03d}"
            wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=run_name,
                config=config,
                reinit=True,
                tags=["hyperparameter_search", f"model_type_{config['model_type']}"]
            )
        
        try:
            # Create model
            num_features = 12  # After dropping column 11
            forecast_horizon = 6
            
            world_model = WorldModel(
                num_features=num_features,
                dim_model=config['dim_model'],
                num_heads=config['num_heads'],
                num_encoder_layers=config['num_encoder_layers'],
                num_decoder_layers=config['num_decoder_layers'],
                encoder_dropout=config['encoder_dropout'],
                decoder_dropout=config['decoder_dropout'],
                max_len=100,
                forecast_horizon=forecast_horizon,
                model_type=config['model_type'],
                device=self.device
            )
            
            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                self.val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False
            )
            test_loader = DataLoader(
                self.test_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False
            )
            
            # Train model
            print("Training model...")
            optimizer = torch.optim.Adam(world_model.model.parameters(), lr=config['learning_rate'])
            criterion = nn.MSELoss()
            
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(config['num_epochs']):
                # Training
                world_model.model.train()
                train_loss = 0.0
                
                for x, pl, y in train_loader:
                    # Ensure consistent float32 dtype
                    x = x.to(self.device, dtype=torch.float32)
                    pl = pl.to(self.device, dtype=torch.float32)
                    y = y.to(self.device, dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    output = world_model.model(x, pl)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * x.size(0)
                
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)
                
                # Validation
                world_model.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for x, pl, y in val_loader:
                        # Ensure consistent float32 dtype
                        x = x.to(self.device, dtype=torch.float32)
                        pl = pl.to(self.device, dtype=torch.float32)
                        y = y.to(self.device, dtype=torch.float32)
                        output = world_model.model(x, pl)
                        loss = criterion(output, y)
                        val_loss += loss.item() * x.size(0)
                
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                
                # Log to wandb
                if WANDB_AVAILABLE and wandb_run:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": min(val_losses)
                    })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = world_model.model.state_dict().copy()
                    if WANDB_AVAILABLE and wandb_run:
                        wandb.log({"new_best_val_loss": best_val_loss})
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{config['num_epochs']} | "
                          f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            # Load best model
            if best_model_state:
                world_model.model.load_state_dict(best_model_state)
            
            print("Evaluating model...")
            
            # Prepare for evaluation
            world_model.mean = self.normalized_data['mean']
            world_model.std = self.normalized_data['std']
            
            # Evaluate model
            metrics = evaluate_model(
                world_model.model,
                test_loader,
                self.normalized_data['mean'],
                self.normalized_data['std'],
                self.normalized_data['raw_test'],
                forecast_horizon=6,
                num_samples=50,
                device=self.device
            )
            
            # Add training metrics
            metrics['train_loss_final'] = train_losses[-1]
            metrics['val_loss_final'] = val_losses[-1]
            metrics['best_val_loss'] = best_val_loss
            
            # Save the trained model
            model_filename = f"run_{run_id:03d}_{config['model_type']}_dim{config['dim_model']}_heads{config['num_heads']}.pth"
            model_path = os.path.join(self.models_dir, model_filename)
            
            try:
                # Save model state dict
                torch.save(world_model.model.state_dict(), model_path)
                metrics['model_path'] = model_path
                print(f"  Model saved to: {model_path}")
                
                # Also save model configuration for easy loading later
                config_filename = f"run_{run_id:03d}_config.json"
                config_path = os.path.join(self.models_dir, config_filename)
                
                model_config = {
                    'run_id': run_id,
                    'model_params': {
                        'num_features': 12,
                        'dim_model': config['dim_model'],
                        'num_heads': config['num_heads'],
                        'num_encoder_layers': config['num_encoder_layers'],
                        'num_decoder_layers': config['num_decoder_layers'],
                        'encoder_dropout': config['encoder_dropout'],
                        'decoder_dropout': config['decoder_dropout'],
                        'max_len': 100,
                        'forecast_horizon': 6,
                        'model_type': config['model_type']
                    },
                    'training_params': {
                        'batch_size': config['batch_size'],
                        'learning_rate': config['learning_rate'],
                        'num_epochs': config['num_epochs']
                    },
                    'metrics': metrics,
                    'model_file': model_filename
                }
                
                with open(config_path, 'w') as f:
                    # Convert numpy values to Python floats for JSON serialization
                    json_config = {}
                    for k, v in model_config.items():
                        if isinstance(v, dict):
                            json_config[k] = {}
                            for k2, v2 in v.items():
                                if isinstance(v2, (np.floating, np.integer)):
                                    json_config[k][k2] = float(v2)
                                elif isinstance(v2, dict):
                                    json_config[k][k2] = {}
                                    for k3, v3 in v2.items():
                                        if isinstance(v3, (np.floating, np.integer)):
                                            json_config[k][k2][k3] = float(v3)
                                        elif isinstance(v3, (np.floating, float)) and np.isnan(v3):
                                            json_config[k][k2][k3] = None
                                        else:
                                            json_config[k][k2][k3] = v3
                                else:
                                    json_config[k][k2] = v2
                        else:
                            json_config[k] = v
                    
                    json.dump(json_config, f, indent=2)
                
                print(f"  Config saved to: {config_path}")
                
            except Exception as e:
                print(f"  Warning: Failed to save model: {e}")
                metrics['model_save_error'] = str(e)
            
            # Log final metrics to wandb
            if WANDB_AVAILABLE and wandb_run:
                final_metrics = {}
                for metric, value in metrics.items():
                    if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                        final_metrics[f"final_{metric}"] = value
                
                wandb.log(final_metrics)
                
                # Log summary metrics
                wandb.run.summary.update({
                    "best_val_loss": best_val_loss,
                    "mae_all_features": metrics.get('mae_all_features', np.nan),
                    "mae_map_only": metrics.get('mae_map_only', np.nan),
                    "trend_accuracy": metrics.get('trend_accuracy', np.nan),
                    "crps": metrics.get('crps', np.nan),
                    "run_id": run_id,
                    "status": "success"
                })
                
                # Mark run as successful
                wandb.log({"run_completed": True})
            
            print(f"Results for run {run_id}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value):
                    print(f"  {metric}: {value:.4f}")
                elif isinstance(value, (int, float, np.floating, np.integer)) and np.isnan(value):
                    print(f"  {metric}: NaN")
                else:
                    print(f"  {metric}: {value}")
            
            return {
                'config': config,
                'metrics': metrics,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error in run {run_id}: {e}")
            
            # Log error to wandb
            if WANDB_AVAILABLE and wandb_run:
                wandb.log({"error": str(e), "run_failed": True})
                wandb.run.summary.update({
                    "status": "failed",
                    "error": str(e),
                    "run_id": run_id
                })
            
            return {
                'config': config,
                'metrics': {},
                'error': str(e),
                'status': 'failed'
            }
        
        finally:
            # Always finish the wandb run
            if WANDB_AVAILABLE and wandb_run:
                wandb.finish()
    
    def run_search(self) -> List[Dict]:
        """Run hyperparameter search."""
        search_space = self.get_hyperparameter_space()
        results = []
        
        print(f"Starting hyperparameter search with {self.max_runs} runs...")
        
        for run_id in range(1, self.max_runs + 1):
            print(f"\n{'='*50}")
            print(f"RUN {run_id}/{self.max_runs}")
            print(f"{'='*50}")
            
            # Sample hyperparameters
            config = self.sample_hyperparameters(search_space)
            
            # Train and evaluate
            result = self.train_and_evaluate(config, run_id)
            results.append(result)
            
            # Save intermediate results
            if run_id % 10 == 0:
                self.save_results(results, f"intermediate_run_{run_id}")
        
        # Save final results
        self.save_results(results, "final")
        
        return results
    
    def save_results(self, results: List[Dict], suffix: str = ""):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = f"hyperparameter_search_{timestamp}_{suffix}.json"
        json_path = os.path.join(self.results_dir, json_filename)
        
        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            json_result = result.copy()
            if 'metrics' in json_result:
                # Convert numpy values to Python floats
                metrics = {}
                for k, v in json_result['metrics'].items():
                    if isinstance(v, (np.floating, np.integer)):
                        metrics[k] = float(v)
                    elif isinstance(v, (float, np.floating)) and np.isnan(v):
                        metrics[k] = None
                    else:
                        metrics[k] = v
                json_result['metrics'] = metrics
            json_results.append(json_result)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save as pickle (preserves exact data types)
        pickle_filename = f"hyperparameter_search_{timestamp}_{suffix}.pkl"
        pickle_path = os.path.join(self.results_dir, pickle_filename)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {json_path} and {pickle_path}")
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze search results."""
        successful_runs = [r for r in results if r['status'] == 'success']
        
        if not successful_runs:
            print("No successful runs found!")
            return {}
        
        print(f"\nAnalysis of {len(successful_runs)} successful runs:")
        
        # Find best configurations for each metric
        metrics = ['mae_all_features', 'mae_map_only', 'trend_accuracy', 'crps']
        best_configs = {}
        
        for metric in metrics:
            valid_runs = [r for r in successful_runs 
                         if metric in r['metrics'] and isinstance(r['metrics'][metric], (int, float, np.floating, np.integer)) and not np.isnan(r['metrics'][metric])]
            
            if not valid_runs:
                continue
            
            if metric == 'trend_accuracy':
                # Higher is better
                best_run = max(valid_runs, key=lambda x: x['metrics'][metric])
            else:
                # Lower is better
                best_run = min(valid_runs, key=lambda x: x['metrics'][metric])
            
            best_configs[metric] = {
                'config': best_run['config'],
                'value': best_run['metrics'][metric]
            }
            
            print(f"\nBest {metric}: {best_run['metrics'][metric]:.4f}")
            print("Config:")
            for k, v in best_run['config'].items():
                print(f"  {k}: {v}")
        
        return best_configs

def main():
    """Main function to run hyperparameter search."""
    # Configuration
    data_path = '/abiomed/downsampled/10min_1hr_all_data.pkl'
    results_dir = "../results"
    models_dir = "/abiomed/hp_search"
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    max_runs = 100
    wandb_project = "timeseries_transformer_search"
    wandb_entity = None  # Set to your wandb username/team if needed
    
    print(f"Running hyperparameter search on {device}")
    print(f"Models will be saved to: {models_dir}")
    if WANDB_AVAILABLE:
        print(f"Logging to wandb project: {wandb_project}")
    
    # Create search object
    searcher = HyperparameterSearch(
        data_path=data_path,
        results_dir=results_dir,
        device=device,
        max_runs=max_runs,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        models_dir=models_dir
    )
    
    # Run search
    results = searcher.run_search()
    
    # Analyze results
    best_configs = searcher.analyze_results(results)
    
    # Create summary wandb run
    if WANDB_AVAILABLE:
        print("\nCreating summary wandb run...")
        summary_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name="search_summary",
            job_type="summary",
            tags=["summary", "hyperparameter_search"]
        )
        
        # Log summary statistics
        successful_runs = [r for r in results if r['status'] == 'success']
        failed_runs = [r for r in results if r['status'] == 'failed']
        
        summary_stats = {
            "total_runs": len(results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "success_rate": len(successful_runs) / len(results) if results else 0
        }
        
        # Add best metric values
        if successful_runs:
            for metric in ['mae_all_features', 'mae_map_only', 'trend_accuracy', 'crps']:
                valid_values = [r['metrics'].get(metric, np.nan) for r in successful_runs 
                               if metric in r['metrics'] and isinstance(r['metrics'][metric], (int, float, np.floating, np.integer)) and not np.isnan(r['metrics'][metric])]
                
                if valid_values:
                    if metric == 'trend_accuracy':
                        summary_stats[f"best_{metric}"] = max(valid_values)
                        summary_stats[f"mean_{metric}"] = np.mean(valid_values)
                    else:
                        summary_stats[f"best_{metric}"] = min(valid_values)
                        summary_stats[f"mean_{metric}"] = np.mean(valid_values)
        
        wandb.log(summary_stats)
        
        # Log best configurations as artifacts
        if best_configs:
            for metric, config_info in best_configs.items():
                wandb.log({f"best_{metric}_value": config_info['value']})
        
        wandb.finish()
    else:
        print("\nSkipping wandb summary (wandb not available)")
    
    # Save analysis
    analysis_filename = f"best_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analysis_path = os.path.join(results_dir, analysis_filename)
    
    with open(analysis_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"\nBest configurations saved to {analysis_path}")
    if WANDB_AVAILABLE:
        print(f"Wandb project: https://wandb.ai/{wandb_entity or 'your-username'}/{wandb_project}")
    print("Hyperparameter search completed!")

if __name__ == "__main__":
    main()