import torch
import numpy as np
def compute_acp_cost(actions):
    acp = 0.0
    for i in range(1, len(actions)):
        acp += np.linalg.norm((actions[i] - actions[i-1]))
        if acp > 1:
            print(f"Change is {acp} when it should be maximum 1")
    return acp

def overall_acp_cost(actions2d, length):
    accumulated_change = 0.0
    for vect in actions2d:
        accumulated_change += compute_acp_cost(vect)
        acp = accumulated_change/length
    return acp

def compute_air_cost(states, actions):
    #take a look in states to see what values there are

    air_multiplier = 1.0
    intensification_needed = 0
    air = 0.0
    i = 0
    for state in range(0, states-1):
        if state<60.0:
            intensification_needed += 1 if actions[i+1] > actions[i]:
            air += 1.0
    i += 1
    return air/len(actions)

