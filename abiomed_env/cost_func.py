import torch
import numpy as np

def compute_acp_cost(actions):
    """Calculates the Action Change Penalty (ACP) for a single episode

    This function iterates through a sequence of actions in an episode and
    sums the change between each consecutive action

    Args:
        actions (list[float] or np.ndarray) is a 1D list or array of
            actions within a single episode

    Returns:
        float: The cumulative action change penalty for the episode
    """
    acp = 0.0
    for i in range(1, len(actions)):
        acp += np.linalg.norm((actions[i] - actions[i-1]))
    return acp

def overall_acp_cost(actions2d):
    """Calculates the mean ACP per timestep

    This function computes the ACP across multiple episodes and normalizes
    it by the total number of timesteps to get a mean value.

    Args:
        actions2d (list[list[float]]) is a 2D list where each inner list is
            the sequence of actions from a single episode

    Returns:
        float: The mean action change penalty per timestep
    """
    total_timesteps = sum(len(episode) for episode in actions2d)
    accumulated_change = 0.0
    for vect in actions2d:
        accumulated_change += compute_acp_cost(vect)
    acp = accumulated_change/total_timesteps
    return acp

def compute_map_air(states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs MAP. Note that each step is an hour.

    Args:
        actions (list[float]) is a list of actions within the episode

        states2D (list[float] is an array of MAP values for the episode
    
    Returns:
        float: total AIR based on MAP for an episode
    """
    opportunities = 0
    correct_intensifications = 0

    for t in range(1, len(states)-1):
        current_state = states[t]

        if current_state < 60.0:
            opportunities += 1
            if (actions[t] > actions[t - 1]):
                correct_intensifications += 1

        elif current_state > 100.0:
            opportunities += 1
            if (actions[t] < actions[t-1]):
                correct_intensifications += 1

    if opportunities == 0:
        return None

    return correct_intensifications / opportunities

def overall_map_cost(actions2d, states2d):
    """
    Calculates the total appropriate intensification rate across all episodes
        based on actions vs MAP

    Args:
        actions2d (list[list[float]]) is a 2D list where each inner list is
            the sequence of actions from a single episode

        states2D (list[list[float]]) is a 2D array where each row is a state
            vector for a single timestep.
    
    Returns:
        float: total AIR based on MAP from all episodes
    """
    all_episode_rates = []
    for episode_states, episode_actions in zip(states2d, actions2d):
        map_vals = [state_vector[0] for state_vector in episode_states]
        rate = compute_map_air(map_vals, episode_actions)
        if rate is not None:
            all_episode_rates.append(rate)

    return np.mean(all_episode_rates)

def compute_hr_air(states, actions):
    opportunities = 0
    correct_intensifications = 0

    for t in range(1, len(states)-1):
        current_state = states[t]

        if current_state <= 50.0:
            opportunities += 1
            if (actions[t] > actions[t - 1]):
                correct_intensifications += 1

        elif current_state >= 100.0:
            opportunities += 1
            if (actions[t] < actions[t-1]):
                correct_intensifications += 1

    if opportunities == 0:
        return None

    return correct_intensifications / opportunities

def overall_hr_cost(actions2d, states2d):
    all_episode_rates = []
    for episode_states, episode_actions in zip(states2d, actions2d):
        map_vals = [state_vector[3] for state_vector in episode_states]
        rate = compute_hr_air(map_vals, episode_actions)
        if rate is not None:
            all_episode_rates.append(rate)

    return np.mean(all_episode_rates)

def compute_pulsatility_air(states, actions):
    opportunities = 0
    correct_intensifications = 0

    for t in range(1, len(states)-1):
        current_state = states[t]

        if current_state <= 10.0:
            opportunities += 1
            if (actions[t] > actions[t - 1]):
                correct_intensifications += 1

    if opportunities == 0:
        return None

    return correct_intensifications / opportunities

def overall_pulsatility_cost(actions2d, states2d):

    all_episode_rates = []
    for episode_states, episode_actions in zip(states2d, actions2d):
        map_vals = [state_vector[7] for state_vector in episode_states]
        rate = compute_pulsatility_air(map_vals, episode_actions)
        if rate is not None:
            all_episode_rates.append(rate)

    return np.mean(all_episode_rates)

#I think I need to look into this b/c not sure if it's the same for all policies
MAP_IDX = 0
HR_IDX = 5
PULSATILITY_IDX = 7

def unstable_states(flattened_states):
    """
    Calculates the percentage of total timesteps that are in an unstable state, 
        which for now is when MAP, HR, or pulsatility are out of the proper range

    Args:
        flattened_states (list[list[float]]): A 2D array where each row is a state
                                     vector for a single timestep.
    """
    total_timesteps = len(flattened_states)
    if total_timesteps == 0:
        return 0.0
        
    unsafe_count = 0
    for state_vector in flattened_states:
        current_map = state_vector[MAP_IDX]
        current_hr = state_vector[HR_IDX]
        current_pulsatility = state_vector[PULSATILITY_IDX]
        is_map_unstable = (current_map < 60.0) or (current_map > 100.0)
        is_hr_unstable = (current_hr <= 50.0) or (current_hr >= 100.0)
        is_pulsatility_unstable = (current_pulsatility <= 10.0)

        if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
            unsafe_count += 1
            
    percentage = (unsafe_count / total_timesteps) * 100
    return percentage