# Time Series Transformer Evaluation Framework

This directory contains a comprehensive evaluation framework for time series transformer models with baseline comparisons and hyperparameter search capabilities. 

Scripts and experiments are mostly for the TS4H submission.

## Overview

The framework implements:
- **Neural Process baseline**: Probabilistic model with latent variable sampling
- **MLP + Dropout baseline**: Simple feedforward network with dropout for uncertainty
- **Comprehensive evaluation metrics**: 6 different metrics including MAE, trend accuracy, and CRPS
- **Hyperparameter search**: Automated search for TimeSeriesTransformer models
- **Data handling**: Proper normalization and column 11 (NaN) handling

## Files Description

### Core Modules
- `data_utils.py`: Data loading, normalization, and utility functions
- `baselines.py`: Neural Process and MLP baseline model implementations  
- `evaluation_metrics.py`: All 6 evaluation metrics implementation
- `hyperparameter_search.py`: Automated hyperparameter search for transformers
- `baseline_evaluation.py`: Main script to evaluate and compare all models

### Test and Documentation
- `test_implementation.py`: Verification script for all components
- `README.md`: This documentation file

## Usage

### 1. Test Implementation (Recommended First Step)
```bash
python test_implementation.py
```
This verifies all components work correctly before running full evaluations.

### 2. Evaluate Baseline Models
```bash
python baseline_evaluation.py
```

This will:
- Train MLP + Dropout baseline
- Train Neural Process baseline  
- Evaluate existing transformer models (if available)
- Compare all models across all 6 metrics
- Save results to `../results/` directory

### 3. Hyperparameter Search for Transformers
```bash
python hyperparameter_search.py
```

This will:
- Run up to 100 hyperparameter configurations
- Search across architecture parameters (dim_model, num_heads, etc.)
- Train and evaluate each configuration
- **Save all trained models to `/abiomed/hp_search/`**
- **Log all runs to Weights & Biases** for real-time monitoring
- Save best configurations and all results
- Take several hours to complete

### 4. Analyze and Load Best Models
```bash
python model_analysis.py
```

This will:
- Analyze all saved models from hyperparameter search
- Create a comprehensive results table with all metrics
- Identify best models for each evaluation metric
- Save detailed CSV report
- Show examples of loading specific models

**Weights & Biases Integration:**
- **Automatic graceful degradation**: Works with or without wandb installed
- Each hyperparameter configuration creates a separate wandb run (if available)
- Real-time logging of training/validation losses during training
- All 6 evaluation metrics logged for each run
- Summary run created at the end with best configurations
- Failed runs are properly logged with error information
- Project name: `timeseries_transformer_search`
- If wandb unavailable: Warning shown, but execution continues normally

## Evaluation Metrics

The framework implements 6 comprehensive metrics:

1. **MAE All Features**: Mean absolute error averaged across all features and time steps
2. **MAP Only MAE**: MAE for Mean Arterial Pressure (feature index 0) only
3. **MAE Static P-level**: MAE for samples where p-level doesn't change
4. **MAE Dynamic P-level**: MAE for samples where p-level changes
5. **Trend Accuracy**: Classification accuracy for MAP trend (up/down/flat)
6. **CRPS**: Continuous Ranked Probability Score with 50 samples

## Data Requirements

- Dataset: `/abiomed/downsampled/10min_1hr_all_data.pkl`
- Shape: `[batch_size, 12, 13]` (12 time steps, 13 features)
- Column 11 contains NaNs and is automatically dropped for baselines
- Input: First 6 time steps + p-levels for last 6 time steps
- Output: All features (except p-level) for last 6 time steps

## Model Architecture Details

### Neural Process Baseline
- Encoder: Processes context sequence with time indices
- Latent space: 128-dimensional with mean/variance parameterization
- Decoder: Generates predictions conditioned on latent and p-levels
- Supports both deterministic and probabilistic inference

### MLP + Dropout Baseline  
- Input: Flattened sequence + p-levels (78 dimensions total)
- Architecture: [512, 256, 128] hidden layers with ReLU and dropout
- Dropout rate: 0.2 for uncertainty estimation
- Output: 66 dimensions (11 features × 6 time steps)

### TimeSeriesTransformer (for search)
- Supports both standard and rotary transformers
- Search space includes: dim_model, num_heads, layers, dropout rates
- Encoder-decoder architecture with positional encoding
- Output dimension: 66 (11 features × 6 time steps)

## Results Format

Results are saved in JSON and pickle formats with structure:
```json
{
  "mlp_dropout": {
    "mae_all_features": 1.23,
    "mae_map_only": 2.34,
    "mae_static": 1.45,
    "mae_dynamic": 1.67,
    "trend_accuracy": 0.78,
    "crps": 0.89,
    "train_loss_final": 0.12,
    "val_loss_final": 0.15,
    "best_val_loss": 0.13
  },
  "neural_process": { ... },
  "transformers": { ... }
}
```

## Hardware Requirements

- GPU: Recommended CUDA device (default: cuda:1)
- Memory: ~8GB GPU memory for larger models
- Time: 
  - Baseline evaluation: ~1-2 hours
  - Hyperparameter search: ~8-12 hours for 100 runs

## Weights & Biases Setup

To use the wandb logging features:

1. **Install wandb** (if not already installed):
```bash
pip install wandb
```

2. **Login to wandb** (first time only):
```bash
wandb login
```

3. **Configure project** (optional):
Edit `hyperparameter_search.py` to change:
- `wandb_project`: Your project name
- `wandb_entity`: Your username or team name

4. **Monitor progress**:
- Visit https://wandb.ai/your-username/timeseries_transformer_search
- View real-time training curves, metrics, and comparisons
- Filter and sort runs by performance

## Notes

- The framework handles device placement automatically
- Models are saved after training for later analysis
- Intermediate results are saved every 10 runs during hyperparameter search
- All metrics handle edge cases (NaN values, empty splits) gracefully
- CRPS calculation uses empirical method with 50 samples

## Troubleshooting

**NumPy compatibility warnings**: These are cosmetic and don't affect functionality. The framework works correctly despite the warnings.

**CUDA out of memory**: Reduce batch size in the evaluation scripts or use CPU by changing device parameter.

**Data loading errors**: Ensure the data file exists at `/abiomed/downsampled/10min_1hr_all_data.pkl` and is accessible.

## Output Directory Structure

```
../results/
├── baseline_evaluation_YYYYMMDD_HHMMSS.json
├── baseline_evaluation_YYYYMMDD_HHMMSS.pkl
├── hyperparameter_search_YYYYMMDD_HHMMSS_final.json
├── hyperparameter_search_YYYYMMDD_HHMMSS_final.pkl
├── best_configs_YYYYMMDD_HHMMSS.json
├── hyperparameter_search_results.csv
├── mlp_baseline_YYYYMMDD_HHMMSS.pth
└── neural_process_baseline_YYYYMMDD_HHMMSS.pth

/abiomed/hp_search/
├── run_001_transformer_dim256_heads8.pth
├── run_001_config.json
├── run_002_rotary_transformer_dim512_heads16.pth
├── run_002_config.json
├── ...
└── run_100_config.json
```

## Model Loading Examples

### Loading the Best Model
```python
from model_analysis import create_detailed_report, find_best_models, load_best_model

# Analyze all results
df = create_detailed_report("/abiomed/hp_search")

# Find best model by MAE all features
best_models = find_best_models(df, 'mae_all_features', top_k=1)
best_model_info = best_models.iloc[0]

# Load the model
world_model = load_best_model(best_model_info, device='cuda:1')

# Use the model for inference
# world_model.load_data("/abiomed/downsampled/10min_1hr_all_data.pkl") 
# predictions = world_model.step(input_sequence, p_level)
```

### Loading by Specific Criteria
```python
# Find best model by trend accuracy
best_trend_models = find_best_models(df, 'trend_accuracy', top_k=3)

# Find models with specific architecture
rotary_models = df[df['model_type'] == 'rotary_transformer']
best_rotary = rotary_models.sort_values('mae_all_features').iloc[0]
```