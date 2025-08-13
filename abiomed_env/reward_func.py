import torch
import numpy as np


def hr_penalty(hr):
    relu = torch.nn.ReLU()
    return relu((hr - 75) ** 2 / 250 - 1).item()


def min_map_penalty(map):
    relu = torch.nn.ReLU()
    return relu(7 * (60 - map) / 20).item()


def pulsat_penalty(pulsat):
    relu = torch.nn.ReLU()
    lower_penalty = relu(7 * (20 - pulsat) / 20).item()
    upper_penalty = relu((pulsat - 50) / 20).item()
    return lower_penalty + upper_penalty


def hypertention_penalty(map):
    relu = torch.nn.ReLU()
    return relu((map - 106) / 18).item()


def compute_reward_smooth(data, map_dim=0, pulsat_dim=6, hr_dim=7, lvedp_dim=3):
    """
    Differentiable version of the reward function using PyTorch
    data: torch.Tensor, shape (batch_size, horizon, num_features)
    """
    score = torch.tensor(0.0, device=data.device)
    # MAP component
    map_data = data[..., map_dim]

    # MinMAP range component
    minMAP = torch.min(map_data)
    score += min_map_penalty(minMAP)

    # hypertention penalty
    meanMAP = torch.mean(map_data)
    score += hypertention_penalty(meanMAP)

    # Heart Rate component
    hr = torch.min(data[..., hr_dim])
    # Polynomial penalty for heart rate outside 50-100 range
    # Quadratic penalty centered at hr=75, max penalty at hr=50 or 100
    score += hr_penalty(hr)

    # Pulsatility component
    pulsat = torch.min(data[..., pulsat_dim])
    score += pulsat_penalty(pulsat)

    return -score


def compute_reward_discrete(data, map_dim=0, pulsat_dim=7, hr_dim=9, lvedp_dim=4):
    score = 0

    ## MAP ##
    map_val = data[..., map_dim]
    if map_val.any() >= 60.0:
        score += 0
    elif (map_val.any() >= 50.0) & (map_val.any() <= 59.0):
        score += 1
    elif (map_val.any() >= 40.0) & (map_val.any() <= 49.0):
        score += 3
    else:
        score += 7  # <40 mmHg

    ## TIME IN HYPOTENSION ##
    ts_hypo = np.mean(map_val < 60) * 100
    if ts_hypo < 0.1:
        score += 0
    elif (ts_hypo >= 0.1) & (ts_hypo <= 0.2):
        score += 1
    elif (ts_hypo > 0.2) & (ts_hypo <= 0.5):
        score += 3
    else:
        score += 7  # greater than 50%

    ## Pulsatility ##
    puls = data[..., pulsat_dim]
    if puls.any() > 20.0:
        score += 0
    elif (puls.any() <= 20.0) & (puls.any() > 10.0):
        score += 5
    else:
        score += 7  # puls <= 10

    ## Heart Rate ##
    hr = data[..., hr_dim]
    if hr.any() >= 100.0:  # tachycardic
        score += 3
    if hr.any() <= 50.0:  # bradycardic
        score += 3

    ## LVEDP ##
    lvedp = data[..., lvedp_dim]
    if lvedp.any() > 20.0:
        score += 7
    if (lvedp.any() >= 15.0) & (hr.any() <= 20.0):
        score += 4

    """
    ## CPO ##
    if cpo > 1.: 
        score+=0
    elif (cpo > 0.6) & (cpo <=1.):
        score+=1
    elif (cpo > 0.5) & (cpo <=0.6):
        score+=3
    else: score+=5 # cpo <=0.5
    """

    return -score
