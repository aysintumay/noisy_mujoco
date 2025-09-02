import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os, sys
# add parent directory to path
sys.path.append("..")

from model import TimeSeriesDataset, TimeSeriesTransformer, TimeSeriesTransformerRotary, WorldModel


def train_model(data_path, model_type, model_path):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    print(f"Loading data from: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print("Data keys:", data.keys())
    print(f"Train data shape: {data['train'].shape}")
    print(f"Val data shape: {data['val'].shape}")
    print(f"Test data shape: {data['test'].shape}")
    print(f"Number of features: {data['train'].shape[2]}")

    # Model parameters
    num_features = 12  # Number of input features
    input_horizon = 6
    forecast_horizon = 6  # Number of time steps to forecast
    output_dim = (num_features - 1) * forecast_horizon  # Exclude p-level from output
        
    if model_type == 'transformer':
        # Create WorldModel with  transformer
        print("\nCreating WorldModel with transformer...")
        wm = WorldModel(
            num_features=num_features,
            dim_model=256,
            num_heads=8,
            num_encoder_layers=3,
            num_decoder_layers=2,
            encoder_dropout=0.1,
            decoder_dropout=0.0,
            max_len=100,
            forecast_horizon=forecast_horizon,
            model_type='transformer',    
            device=device
        )
    elif model_type == 'rotary_transformer':
        # Create WorldModel with rotary transformer
        print("\nCreating WorldModel with rotary transformer...")
        wm = WorldModel(
            num_features=num_features,
            dim_model=256,
            num_heads=8,
            num_encoder_layers=3,
            num_decoder_layers=2,
            encoder_dropout=0.1,
            decoder_dropout=0.0,
            max_len=100,
            forecast_horizon=forecast_horizon,
            model_type='rotary_transformer',  # Use rotary transformer
            device=device
        )
        
    print(f"Model created with {num_features} input features")
    print(f"Input horizon: {input_horizon}, Forecast horizon: {forecast_horizon}")

    # Load data into the model
    print("\nLoading data into the model...")
    wm.load_data(data_path)
    print("Data loaded successfully!")

    # Training parameters
    num_epochs = 35
    batch_size = 64
    learning_rate = 0.001

    print(f"\nStarting training with:")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # Train the model
    print("\nTraining the model...")
    best_model = wm.train_model(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    print("\nTraining completed!")

    # Test the trained model
    print("\nTesting the trained model...")
    test_mse, test_mape = wm.test(loss_fn='mae')
    print(f"Final Test MAE: {test_mse:.6f}")
    print(f"Final Test MAPE: {test_mape:.3f}")

    # Save the trained model
    print(f"\nSaving model to: {model_path}")
    wm.save_model(model_path)
    print(f"Model saved successfully!")

def main():
    data_path = '/abiomed/downsampled/10min_1hr_all_data_subsampled.pkl'
    
    model_type = 'transformer'
    for i in range(1, 4):
        model_path = f'/abiomed/downsampled/models/transformer_1hr_subsampled_new{i}.pth'
        train_model(data_path, model_type, model_path)
    
    model_type = 'rotary_transformer'
    for i in range(1, 4):
        model_path = f'/abiomed/downsampled/models/rotary_1hr_subsampled_new{i}.pth'
        train_model(data_path, model_type, model_path)


if __name__ == "__main__":
    main()