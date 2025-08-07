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



# def compute_air_cost(states, actions):
#     """Calculates the total penalty due to Appropriate Intensification Rate (AIR) over an episode
#     Args:
#         states ()

#     """
#     #map, pulsatility, hr
#     map <40 or >60

#     if ()
#     intensification_needed = 0
#     air = 0.0
#     i = 0
#     for state in range(0, states-1):
#         if state<60.0:
#             intensification_needed += 1 if actions[i+1] > actions[i]:
#             air += 1.0
#     i += 1
#     return air/len(actions)


#for the doctor policy may do something like acp to include between episodes
def compute_map_air(states, actions):
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(states)-2):
        current_state = states[t]

        if current_state < 60:
            opportunities += 1
            if (actions[t] > actions[t - 1]):
                correct_intensifications += 1

        elif current_state > 100:
            opportunities += 1
            if (actions[t] < actions[t-1]):
                correct_intensifications += 1

    if opportunities == 0:
        return None

    return correct_intensifications / opportunities




def overall_air_cost(actions2d, states2d):
#map is the first column in states
    all_episode_rates = []
    for episode_states, episode_actions in zip(states2d, actions2d):
        map_vals = [state_vector[0] for state_vector in episode_states]
        rate = compute_map_air(map_vals, episode_actions)
        if rate is not None:
            all_episode_rates.append(rate)

    return np.mean(all_episode_rates)
