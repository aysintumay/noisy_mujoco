import torch



model_kwargs_10min_2hr_window = {
    'num_features': 12,
    'dim_model': 256,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
}

# 5min_1hr_window
model_kwargs_5min_1hr_window = {
    'num_features': 12,
    'forecast_horizon': 12,
    'dim_model': 512,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
}

# 5 min 2 hr window
model_kwargs_5min_2hr_window = {
    'num_features': 12,
    'forecast_horizon': 24,
    'dim_model': 512,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
}

# 10 min 1 hr window
model_kwargs_10min_1hr_window = {
    'num_features': 12,
    'forecast_horizon': 6,
    'dim_model': 128,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
}

model_kwargs_10min_1hr_col = {
    'model_name': "10min_1hr_window_9feat",
    'num_features': 9,
    'forecast_horizon': 6,
    'dim_model': 128,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    # 'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'columns': [0, 3, 4, 5, 6, 7, 9, 10, 12]
}

model_kwargs_10min_1hr_9col2 = {
#    'model_name': "10min_1hr_window_9feat_model2",
    'num_features': 9,
    'forecast_horizon': 6,
    'dim_model': 128,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'encoder_dropout': 0.1,
    'decoder_dropout': 0,
    'max_len': 100,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'columns': [0, 3, 4, 5, 6, 7, 9, 10, 12]
}


model_configs = {
    "10min_1hr_window": model_kwargs_10min_1hr_window,
    "10min_2hr_window": model_kwargs_10min_2hr_window,
    "5min_1hr_window": model_kwargs_5min_1hr_window,
    "5min_2hr_window": model_kwargs_5min_2hr_window,
}