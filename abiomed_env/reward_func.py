import torch
import numpy as np
from cost_func import (aggregate_air_model,
                        weaning_score_model,
                        aggregate_air_physician,
                        weaning_score_physician,
                        compute_acp_cost,
                       
                       )

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


def compute_reward_smooth(data, map_dim=0, pulsat_dim=7, hr_dim=9, lvedp_dim=4):
    """
    Differentiable version of the reward function using PyTorch
    data: torch.Tensor, shape (batch_size, horizon, num_features)
    """
    assert data.shape[-1] == 12, "data should have 12 features"
    assert data.shape[-2] == 6, "data should have 6 time steps"

    score = torch.tensor(0.0, device=data.device)
    # MAP component
    map_data = data[..., map_dim]
    

    # MinMAP range component
    minMAP = torch.min(map_data)
    if minMAP.reshape(-1) < 10:
        raise ValueError("MAP must be unnormalized before reward calculation")
    score += min_map_penalty(minMAP)

    # hypertention penalty
    meanMAP = torch.mean(map_data)
    score += hypertention_penalty(meanMAP)

    # Heart Rate component
    hr = torch.min(data[..., hr_dim])
    if hr.reshape(-1) < 10:
        raise ValueError("HR must be unnormalized before reward calculation")
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


def compute_shaped_reward(data, actions, gamma1, gamma2, gamma3):
    """
    Computes reward score with the addition of ACP, WS, and AIR costs for one sample. 
    data: torch.Tensor, shape (1, 6, num_features), normalized data
    actions: torch.Tensor, shape (1, 2), unnormalized actions
    gamma1, gamma2, gamma3: float, weighting factors for the cost components
    returns: np.float64, final reward
    """

    acp = compute_acp_cost(actions,data) #max 8, min 0 
    air = aggregate_air_physician(data, actions) #max 1 min 0
    ws = weaning_score_physician(data, actions) #max 2, min -1
    
    final_rwd = gamma2*ws + gamma3*air - gamma1 *acp 
    return np.float64(final_rwd)

