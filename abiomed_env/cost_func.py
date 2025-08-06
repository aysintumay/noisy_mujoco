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

def overall_acp_cost(actions2d, length):
    """Calculates the mean ACP per timestep

    This function computes the ACP across multiple episodes and normalizes
    it by the total number of timesteps to get a mean value.

    Args:
        actions2d (list[list[float]]) is a 2D list where each inner list is
            the sequence of actions from a single episode
        total_timesteps (int) is the total number of steps taken across all
            the episodes combined

    Returns:
        float: The mean action change penalty per timestep
    """
    accumulated_change = 0.0
    for vect in actions2d:
        accumulated_change += compute_acp_cost(vect)
        acp = accumulated_change/length
    return acp

#if conditions bad, increase treatment
#if conditions good, don't change treatment

# def compute_air_cost(states, actions):
#     #take a look in states to see what values there are

#     air_multiplier = 1.0
#     intensification_needed = 0
#     air = 0.0
#     i = 0
#     for state in range(0, states-1):
#         if state<60.0:
#             intensification_needed += 1 if actions[i+1] > actions[i]:
#             air += 1.0
#     i += 1
#     return air/len(actions)

