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

def compute_map_model_air(world_model,states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs MAP. Note that 6 states make up an hour.

    Args:
        world_model (object) is an instance of the world_model used to unnormalize
            based on the mean and std.
        states (list[list[float]]) is a list of normalized state vectors for the
            episode and the first column is MAP
        actions (list[float]) is a list of p-levels

    Returns:
        float: The calculated AIR between 0.0 and 1.0 and just 0.0
               if no opportunities for intensification occurred
    """
    #Unnormalize the map column
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    map_values = world_model.unnorm_state_col(col_idx=0, state_vectors=reshaped_states)

    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(map_values)):
        if np.min(map_values[t]) < 60.0:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1

    if opportunities == 0:
        return 0.0
        
    return correct_intensifications / opportunities


def compute_map_physician_air(states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs MAP. Note that each step is an hour.

    Each episode has T timesteps in the state. Each action is one value per state

    Args:
        actions (list[float]) is a list of actions within the episode

        states (list[float]) is a list of unnormalized states of MAP values for the episode
    
    Returns:
        float: total AIR based on MAP for an episode
    """

    avg_map_values = []
    sampled_actions = []
    for i in range(0, len(states) - 5, 6):
        map_chunk = states[i : i+6]
        avg_map_values.append(min(map_chunk))
        sampled_actions.append(actions[i+5])

    opportunities = 0
    correct_intensifications = 0

    for t in range(1, len(avg_map_values)):
        if avg_map_values[t] < 60.0:
            opportunities += 1
            if sampled_actions[t] > sampled_actions[t - 1]:
                correct_intensifications += 1

    if opportunities == 0:
        return 0.0
        
    return correct_intensifications / opportunities



def compute_hr_model_air(world_model, states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs HR. Note that 6 states make up an hour.

    Args:
        world_model (object) is an instance of the world_model used to unnormalize
            based on the mean and std.
        states (list[list[float]]) is a list of normalized state vectors for the
            episode and the 9th column is HR
        actions (list[float]) is a list of p-levels

    Returns:
        float: The calculated AIR between 0.0 and 1.0 and just 0.0
               if no opportunities for intensification occurred
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    hr_values = world_model.unnorm_state_col(col_idx=9, state_vectors=reshaped_states)
    

    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(hr_values)):
        if np.min(hr_values[t]) <= 50.0:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1


    if opportunities == 0:
        return 0.0
        
    return correct_intensifications / opportunities



def compute_hr_physician_air(states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs MAP. Note that each step is an hour.

    Each episode has T timesteps in the state. Each action is one value per state

    Args:
        actions (list[float]) is a list of actions within the episode

        states (list[float]) is a list of unnormalized states of MAP values for the episode
    
    Returns:
        float: total AIR based on MAP for an episode
    """
    avg_hr_values = []
    sampled_actions = []
    for i in range(0, len(states) - 5, 6):
        hr_chunk = states[i : i+6]
        avg_hr_values.append(min(hr_chunk))
        sampled_actions.append(actions[i+5])

    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(avg_hr_values)):
        if avg_hr_values[t] <= 50.0:
            opportunities += 1
            if sampled_actions[t] > sampled_actions[t - 1]:
                correct_intensifications += 1
        # elif avg_hr_values[t] >= 100.0:
        #     opportunities += 1
        #     if sampled_actions[t] < sampled_actions[t - 1]:
        #         correct_intensifications += 1

    if opportunities == 0:
        return 0.0
        
    return correct_intensifications / opportunities

def compute_pulsatility_physician_air(states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs MAP. Note that each step is an hour.

    Each episode has T timesteps in the state. Each action is one value per state

    Args:
        actions (list[float]) is a list of actions within the episode

        states (list[float]) is a list of unnormalized states of MAP values for the episode
    
    Returns:
        float: total AIR based on MAP for an episode
    """
    avg_pulsatility_values = []
    sampled_actions = []
    for i in range(0, len(states) - 5, 6):
        pulsatility_chunk = states[i : i+6]
        avg_pulsatility_values.append(min(pulsatility_chunk))
        sampled_actions.append(actions[i+5])

    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(avg_pulsatility_values)):
        if avg_pulsatility_values[t] <= 20.0:
            opportunities += 1
            if sampled_actions[t] > sampled_actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 0.0
    return correct_intensifications / opportunities

def compute_pulsatility_model_air(world_model, states, actions):
    """
    Calculates the total appropriate intensification rate across a single episode
        based on actions vs pulsatility. Note that 6 states make up an hour.

    Args:
        world_model (object) is an instance of the world_model used to unnormalize
            based on the mean and std.
        states (list[list[float]]) is a list of normalized state vectors for the
            episode and the 7th column is pulsatility
        actions (list[float]) is a list of p-levels

    Returns:
        float: The calculated AIR between 0.0 and 1.0 and just 0.0
               if no opportunities for intensification occurred
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    pulsatility_values = world_model.unnorm_state_col(col_idx=7, state_vectors=reshaped_states)

    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(pulsatility_values)):
        if np.min(pulsatility_values[t]) <= 20.0:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1

    if opportunities == 0:
        return 0.0
        
    return correct_intensifications / opportunities
#indices from the readme 
MAP_IDX = 0
HR_IDX = 9
PULSATILITY_IDX = 7

def is_stable(states):
    """
    Checks if a 1 hour window which is 6 steps is stable.

    Args:
        states (list[list[float]]): A list with exactly 6 state
            vectors with the different features

    Returns:
        bool: Returns True if the hour is stable and False if not
    """
    assert len(states) == 6, "There are not 6 timesteps"
    hour_np = np.array(states)
    map_values = hour_np[:, MAP_IDX]
    hr_values = hour_np[:, HR_IDX]
    pulsatility_values = hour_np[:, PULSATILITY_IDX]

    is_map_unstable = min(map_values) < 60.0
    is_hr_unstable = (min(hr_values) <= 50.0) or (max(hr_values) >= 100.0)
    is_pulsatility_unstable = min(pulsatility_values) <= 10.0

    if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
        return False
    return True

def unstable_percentage(flattened_states):
    """
    Calculates the percentage of total timesteps that are in an unstable state, 
        which for now is when MAP, HR, or pulsatility are out of the proper range

    Args:
        flattened_states (list[list[float]]): A 2D array where each row is a state
                                     vector for a single timestep.

    Returns:
        percentage (float) is the percentage of unsafe states
    """
    unstable_hour_count = 0
    total_hours = 0
    for i in range(0, len(flattened_states) - 5, 6):
        total_hours += 1
        hour_chunk_np = np.array(flattened_states[i : i+6])
        map_in_hour = hour_chunk_np[:, MAP_IDX]
        hr_in_hour = hour_chunk_np[:, HR_IDX]
        pulsatility_in_hour = hour_chunk_np[:, PULSATILITY_IDX]

        is_map_unstable = (map_in_hour < 60).any() 
        is_hr_unstable = (hr_in_hour < 50).any()
        is_pulsatility_unstable = (pulsatility_in_hour <= 10).any()


        if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
            unstable_hour_count += 1
    if total_hours == 0:
        return 0.0
            
    percentage = (unstable_hour_count / total_hours) * 100
    return percentage

def unstable_percentage_model(world_model, states):
    """
    Calculates the percentage of total timesteps that are in an unstable state, 
        which for now is when MAP, HR, or pulsatility are out of the proper range

    Args:
        states (list[list[float]]): A 2D array where each row is a state
                                     vector for a single timestep.

    Returns:
        percentage (float) is the percentage of unsafe states
    """
    unstable_hour_count = 0
    total_hours = 0
    reshaped_states = states.reshape(len(states), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_state_vectors(reshaped_states)
    for i in range(0, len(unnormalized_states)):
        total_hours += 1
        current_hour_data = unnormalized_states[i]
        map_in_hour = current_hour_data[:,MAP_IDX]
        hr_in_hour = current_hour_data[:,HR_IDX]
        pulsatility_in_hour = current_hour_data[:,PULSATILITY_IDX]

        is_map_unstable = (map_in_hour < 60).any() 
        is_hr_unstable = (hr_in_hour < 50).any()
        is_pulsatility_unstable = (pulsatility_in_hour <= 10).any()


        if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
            unstable_hour_count += 1
    if total_hours == 0:
        return 0.0
            
    percentage = (unstable_hour_count / total_hours) * 100
    return percentage


def weaning_score_physician(flattened_states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        actions (list[float]): A list of p levels for each state.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """
    hourly_states_list = []
    hourly_actions = []
    for i in range(0, len(actions) - 5, 6):
        hour_chunk = flattened_states[i : i+6]
        hourly_states_list.append(hour_chunk)
        hourly_actions.append(actions[i+5])
        
    score = 0.0
    denom = 0.0
    for t in range(1, len(hourly_actions)):
        if is_stable(hourly_states_list[t]):
            denom += 1.0
            current_action = hourly_actions[t]
            previous_action = hourly_actions[t-1]
            increase_diff = current_action - previous_action
            if (previous_action-current_action) == 1:
                score += 1.0
            
            elif increase_diff > 0:
                score -= increase_diff

    return score / denom if denom != 0 else 0.0

def weaning_score_model(world_model, states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        actions (list[float]): A list of p levels for each state.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """

    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_state_vectors(reshaped_states)
    score = 0.0
    denom = 0.0
    for t in range(1, len(actions)):
        if is_stable(unnormalized_states[t]):
            denom += 1.0
            current_action = actions[t]
            previous_action = actions[t-1]
            increase_diff = current_action - previous_action
            if (previous_action-current_action) == 1:
                score += 1.0
            
            elif increase_diff > 0:
                score -= increase_diff

    return score / denom if denom != 0 else 0.0

#note that like the other air functions, the adding of episode airs gets done outside
def aggregate_air_physician(states, actions):
    """
    Calculates the total AIR using the is_stable function.

    Args:
        states (list[list[float]]): A 2D array of state vectors.
        actions (list[float]): A list of actions corresponding to the states.

    Returns:
        float: The AIR score from 0.0 to 1.0
    """
    hourly_states_list = []
    hourly_actions = []
    for i in range(0, len(actions) - 5, 6):
        hourly_states_list.append(states[i : i+6])
        hourly_actions.append(actions[i+5])

    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(hourly_actions)):
        if not is_stable(hourly_states_list[t]):
            opportunities += 1
            if hourly_actions[t] > hourly_actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 0.0
    return correct_intensifications / opportunities

def aggregate_air_model(world_model,states, actions):
    """
    Calculates the total AIR using the is_stable function.

    Args:
        states (list[list[float]]): A 2D array of state vectors.
        actions (list[float]): A list of actions corresponding to the states.

    Returns:
        float: The AIR score from 0.0 to 1.0
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_state_vectors(reshaped_states)
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(actions)):
        if not is_stable(unnormalized_states[t]):
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
    return correct_intensifications / opportunities