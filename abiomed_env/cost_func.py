import torch
import numpy as np

def compute_acp_cost(actions, states):
    """Calculates the Action Change Penalty (ACP) for a single episode

    This function iterates through a sequence of actions in an episode and
    sums the change between each consecutive action

    Args:
        actions (list[float] or np.ndarray) is a 1D list or array of
            actions within a single episode

    Returns:
        float: The cumulative action change penalty for the episode
    """
    reshaped_states = states.reshape(-1, 6, 12)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(reshaped_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    acp = 0.0
    for i in range(1, len(all_actions)):
        if np.linalg.norm((all_actions[i] - all_actions[i-1])) > 2:
            acp += np.linalg.norm((all_actions[i] - all_actions[i-1]))
    return acp

def compute_acp_cost_model(world_model, actions, states):
    """Calculates the Action Change Penalty (ACP) for a single episode

    This function iterates through a sequence of actions in an episode and
    sums the change between each consecutive action

    Args:
        actions (list[float] or np.ndarray) is a 1D list or array of
            actions within a single episode

    Returns:
        float: The cumulative action change penalty for the episode
    """
    reshaped_states = states.reshape(-1, world_model.forecast_horizon, 12)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(unnormalized_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    acp = 0.0
    for i in range(1, len(all_actions)):
        if np.linalg.norm((all_actions[i] - all_actions[i-1])) > 2:
            acp += np.linalg.norm((all_actions[i] - all_actions[i-1]))
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

def super_metric(world_model, states, actions):
    """
    Calculates the Action Change Penalty (ACP) for a single episode
        if weaning is not succesful.

    Args:
        actions (list[float] or np.ndarray) is a 1D list or array of
            actions within a single episode
        states (list[list[float]]): A 2D array of state vectors for the episode.
        world_model (object): An instance of the world model used to unnormalize

    Returns:
        float: The cumulative action change penalty for the episode
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_state_vectors(reshaped_states)
    acp = 0.0
    for t in range(1, len(actions)):
        if is_stable(unnormalized_states[t-1]) and (actions[t] - actions[t-1]) >= 1:
            acp += np.linalg.norm((actions[t] - actions[t-1]))
        if not is_stable(unnormalized_states[t-1]) and (actions[t] - actions[t-1]) < 1:
            acp += np.linalg.norm((actions[t] - actions[t-1]))
        if not is_stable(unnormalized_states[t-1]) and (actions[t] - actions[t-1]) >= 1:
            acp -= np.linalg.norm((actions[t] - actions[t-1]))

    return acp

def is_stable_1d(state_vector):
    #to match acp
    map_value = state_vector[MAP_IDX]
    hr_value = state_vector[HR_IDX]
    pulsatility_value = state_vector[PULSATILITY_IDX]
    is_map_unstable = map_value < 60.0
    is_hr_unstable = (hr_value < 60.0)
    is_pulsatility_unstable = pulsatility_value < 0.3
    return not (is_map_unstable or is_hr_unstable or is_pulsatility_unstable)

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
        if np.min(map_values[t-1]) < 60.0:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1

    if opportunities == 0:
        return 1.0
        
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
        if avg_map_values[t-1] < 60.0:
            opportunities += 1
            if sampled_actions[t] > sampled_actions[t - 1]:
                correct_intensifications += 1

    if opportunities == 0:
        return 1.0
        
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
        if np.min(hr_values[t-1]) <= 50.0:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1


    if opportunities == 0:
        return 1.0
        
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
        if avg_hr_values[t-1] <= 50.0:
            opportunities += 1
            if sampled_actions[t] > sampled_actions[t - 1]:
                correct_intensifications += 1
        # elif avg_hr_values[t] >= 100.0:
        #     opportunities += 1
        #     if sampled_actions[t] < sampled_actions[t - 1]:
        #         correct_intensifications += 1

    if opportunities == 0:
        return 1.0
        
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
        if avg_pulsatility_values[t-1] <= 20.0:
            opportunities += 1
            if sampled_actions[t] > sampled_actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
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
        if np.min(pulsatility_values[t-1]) <= 20.0:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1

    if opportunities == 0:
        return 1.0
        
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
    # assert len(states) == 6, "There are not 6 timesteps"
    hour_np = np.array(states)
    map_values = hour_np[..., MAP_IDX]
    hr_values = hour_np[..., HR_IDX]
    pulsatility_values = hour_np[..., PULSATILITY_IDX]

    is_map_unstable = min(map_values) < 60.0
    is_hr_unstable = (min(hr_values) <= 50.0) # or (max(hr_values) >= 100.0)
    is_pulsatility_unstable = min(pulsatility_values) <= 20.0

    if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
        return False
    return True

arbitrary_threshold = -0.25
def is_stable_gradient(states):
    """
    Checks if a 1 hour window which is 6 steps is stable using the definition that 
    MAP, HR, Pulsatility gradients are not below a threshold

    Args:
        states (list[list[float]]): A list with exactly 6 state
            vectors with the different features

    Returns:
        bool: Returns True if the hour is stable and False if not
    """
    states_np = np.array(states)
    x_vals = np.arange(len(states_np))

    map_values = states_np[:, MAP_IDX]
    hr_values = states_np[:, HR_IDX]
    pulsatility_values = states_np[:, PULSATILITY_IDX]

    map_slope = np.polyfit(x_vals, map_values, 1)[0]
    hr_slope = np.polyfit(x_vals, hr_values, 1)[0]
    pulsatility_slope = np.polyfit(x_vals, pulsatility_values, 1)[0]
    is_map_unstable = abs(map_slope) >= -arbitrary_threshold
    is_hr_unstable = abs(hr_slope) >= -arbitrary_threshold
    is_pulsatility_unstable = abs(pulsatility_slope) >= -arbitrary_threshold

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
    # hourly_states_list = []
    # hourly_actions = []
    # for i in range(0, len(actions) - 5, 6):
    #     hour_chunk = flattened_states[i : i+6]
    #     hourly_states_list.append(hour_chunk)
    #     hourly_actions.append(actions[i+5])
    reshaped_states = flattened_states.reshape(-1, 6, 12)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(reshaped_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    score = 0.0
    denom = 0.0
    for t in range(1, len(all_actions)):
        if is_stable_gradient(flattened_states[t-1]):
            denom += 1.0
            current_action = all_actions[t]
            previous_action = all_actions[t-1]
            increase_diff = current_action - previous_action
            if ((previous_action-current_action) == 1) or ((previous_action-current_action) == 2) :
                score += previous_action-current_action
            
            elif increase_diff > 0:
                score -= 1

    return score / denom if denom != 0 else 0.0

#change is_stable to be within range of the threshold += T
def weaning_score_model(world_model, states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        action (list[float]): A list of p levels for each state, unnormalized.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """

    reshaped_states = states.reshape(-1, world_model.forecast_horizon, 12)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(unnormalized_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    score = 0.0
    denom = 0.0
    for t in range(1, len(all_actions)):
        if is_stable_gradient(unnormalized_states[t-1]):
            denom += 1.0
            current_action = all_actions[t]
            previous_action = all_actions[t-1]
            increase_diff = current_action - previous_action
            if ((previous_action-current_action) == 1) or ((previous_action-current_action) == 2) :
                score += (previous_action-current_action)
            
            elif increase_diff > 0:
                score -= 1

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
    # hourly_states_list = []
    # hourly_actions = []
    # for i in range(0, len(actions) - 5, 6):
    #     hourly_states_list.append(states[i : i+6])
    #     hourly_actions.append(actions[i : i+6])
    reshaped_states = states.reshape(-1, 6, 12)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(reshaped_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    opportunities = 0
    correct_intensifications = 0
    
    for t in range(1, len(all_actions)):
        if not is_stable(states[t-1]):
            
            opportunities += 1
            if all_actions[t] > all_actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
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
    reshaped_states = states.reshape(-1, world_model.forecast_horizon, 12)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(unnormalized_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(all_actions)):
        if not is_stable(unnormalized_states[t-1]):
            opportunities += 1
            if all_actions[t] > all_actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
    return correct_intensifications / opportunities

def compute_air_map_gradient_threshold(world_model, states, actions):
    """Calculates physician AIR score for MAP.

    The score considers cases when the metric is below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The AIR score for MAP for the length of states, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    map_values = unnormalized_states[:, :, 0]

    air_value = 0.0
    x_vals = np.arange(world_model.forecast_horizon)
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(actions)):
        slope = np.polyfit(x_vals, map_values[t-1], 1)[0]
        if slope < arbitrary_threshold:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
    return correct_intensifications / opportunities
        
def compute_air_hr_gradient_threshold(world_model, states, actions):
    """Calculates physician AIR score for HR.

    The score considers cases when the metric is below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The AIR score for HR for the length of states, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    hr_values = unnormalized_states[:, :, 9]

    air_value = 0.0
    x_vals = np.arange(world_model.forecast_horizon)
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(actions)):
        slope = np.polyfit(x_vals, hr_values[t-1], 1)[0]
        if slope < arbitrary_threshold:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
    return correct_intensifications / opportunities
        
def compute_air_pulsat_gradient_threshold(world_model, states, actions):
    """Calculates physician AIR score for pulsatility.

    The score considers cases when the metric is below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The AIR score for pulsatility for the length of states, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    pulsat_values = unnormalized_states[:, :, 7]

    air_value = 0.0
    x_vals = np.arange(world_model.forecast_horizon)
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(actions)):
        slope = np.polyfit(x_vals, pulsat_values[t-1], 1)[0]
        if slope < arbitrary_threshold:
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
    return correct_intensifications / opportunities

#if all slopes are downwards (rather than the mean slope (may need different thresholds))
def compute_air_aggregate_gradient_threshold(world_model, states, actions):
    """Calculates a cumulative aggregate AIR score.

    The score considers cases when all three metrics' gradients are below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The cumulative aggregate AIR score, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    map_values = unnormalized_states[:, :, 0]
    hr_values = unnormalized_states[:, :, 9]
    pulsat_values = unnormalized_states[:, :, 7]

    air_value = 0.0
    x_vals = np.arange(world_model.forecast_horizon)
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(actions)):
        slope1 = np.polyfit(x_vals, map_values[t-1], 1)[0]
        slope2 = np.polyfit(x_vals, hr_values[t-1], 1)[0]
        slope3 = np.polyfit(x_vals, pulsat_values[t-1], 1)[0]
        if (slope1 < arbitrary_threshold) and (slope2 < arbitrary_threshold) and (slope3 < arbitrary_threshold):
            opportunities += 1
            if actions[t] > actions[t - 1]:
                correct_intensifications += 1
    if opportunities == 0:
        return 1.0
    return correct_intensifications / opportunities

#below functions multiply action change by gradient
def compute_map_air_gradient(world_model, states, actions):
    """Calculates an AIR score for MAP.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken.

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The AIR score for MAP for the length of states, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    map_values = unnormalized_states[:, :, 0]
    air_value = 0
    x_vals = np.arange(world_model.forecast_horizon)
    for t in range(1, len(map_values)):
        slope = (np.polyfit(x_vals, map_values[t-1], 1))[0]
        #slope times change in action
        air_value += max(0,(-slope))*max(0,(actions[t]-actions[t-1]))
    return air_value

def compute_hr_air_gradient(world_model, states, actions):
    """Calculates an AIR score for HR.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken.

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The AIR score for HR for the length of states, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    hr_values = unnormalized_states[:, :, 9]
    air_value = 0
    x_vals = np.arange(world_model.forecast_horizon)
    for t in range(1, len(hr_values)):
        slope = (np.polyfit(x_vals, hr_values[t-1], 1))[0]
        air_value += max(0,(-slope))*max(0,(actions[t]-actions[t-1]))
    return air_value

def compute_pulsat_air_gradient(world_model, states, actions):
    """Calculates an AIR score for pulsatility.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken.

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The AIR score for pulsatility for the length of states, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    pulsat_values = unnormalized_states[:, :, 7]
    air_value = 0
    x_vals = np.arange(world_model.forecast_horizon)
    for t in range(1, len(pulsat_values)):
        slope = (np.polyfit(x_vals, pulsat_values[t-1], 1))[0]
        air_value += max(0,(-slope))*max(0,(actions[t]-actions[t-1]))
    return air_value

def compute_aggregate_air_gradient(world_model, states, actions):
    """Calculates a cumulative aggregate AIR score.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken. The gradient is determined
    as the mean of MAP, HR, and Pulsatility gradients

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The cumulative aggregate AIR score, which is a float
    """
    reshaped_states = states.reshape(len(actions), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    map_values = unnormalized_states[:, :, 0]
    hr_values = unnormalized_states[:, :, 9]
    pulsat_values = unnormalized_states[:, :, 7]

    air_value = 0
    x_vals = np.arange(world_model.forecast_horizon)
    for t in range(1, len(actions)):
        slope_map = np.polyfit(x_vals, map_values[t-1], 1)[0]
        slope_hr = np.polyfit(x_vals, hr_values[t-1], 1)[0]
        slope_pulsat = np.polyfit(x_vals, pulsat_values[t-1], 1)[0]
        mean_slope = (slope_map + slope_hr + slope_pulsat) / 3.0
        action_change = actions[t] - actions[t-1]
        air_value += max(0, -mean_slope) * max(0, action_change)
        
    return air_value

FORECAST_HORIZON = 6

def compute_map_air_gradient_threshold_physician(states, actions):
    """Calculates physician AIR score for MAP

    The score considers cases when the metric is below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array
        actions is a 1D array

    Returns:
        The AIR score for the metric for the length of states, which is a float
    """
    x_vals = np.arange(FORECAST_HORIZON)
    opportunities, correct_intensifications = 0, 0
    for t in range(1, len(actions)):
        window_states = states[t-1]
        map_window = window_states[:, MAP_IDX]
        slope = np.polyfit(x_vals, map_window, 1)[0]
        if slope < arbitrary_threshold:
            opportunities += 1
            if actions[t] > actions[t-1]:
                correct_intensifications += 1
    return correct_intensifications / opportunities if opportunities > 0 else 1.0

def compute_hr_air_gradient_threshold_physician(states, actions):
    """Calculates physician AIR score for HR

    The score considers cases when the metric is below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array
        actions is a 1D array

    Returns:
        The AIR score for the metric for the length of states, which is a float
    """
    x_vals = np.arange(FORECAST_HORIZON)
    opportunities, correct_intensifications = 0, 0
    for t in range(1, len(actions)):
        window_states = states[t-1]
        hr_window = window_states[:, HR_IDX]
        slope = np.polyfit(x_vals, hr_window, 1)[0]
        if slope < arbitrary_threshold:
            opportunities += 1
            if actions[t] > actions[t-1]:
                correct_intensifications += 1
    return correct_intensifications / opportunities if opportunities > 0 else 1.0

def compute_pulsat_air_gradient_threshold_physician(states, actions):
    """Calculates physician AIR score for Pulsatility

    The score considers cases when the metric is below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array
        actions is a 1D array

    Returns:
        The AIR score for the metric for the length of states, which is a float
    """
    x_vals = np.arange(FORECAST_HORIZON)
    opportunities, correct_intensifications = 0, 0
    for t in range(1, len(actions)):
        window_states = states[t-1] # Correctly select the (t-1)th hour
        pulsat_window = window_states[:, PULSATILITY_IDX]
        slope = np.polyfit(x_vals, pulsat_window, 1)[0]
        if slope < arbitrary_threshold:
            opportunities += 1
            if actions[t] > actions[t-1]:
                correct_intensifications += 1
    return correct_intensifications / opportunities if opportunities > 0 else 1.0


#if all slopes are downwards (rather than the mean slope (may need different thresholds))
def compute_air_aggregate_gradient_threshold_physician(states, actions):
    """Calculates a cumulative aggregate physician AIR score.

    The score considers cases when all three metrics' gradients are below a 
    threshold as an opportunity and a subsequent increase
    in action as a correct intensification

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The cumulative aggregate AIR score, which is a float
    """
    x_vals = np.arange(FORECAST_HORIZON)
    opportunities = 0
    correct_intensifications = 0
    for t in range(1, len(actions)):
        state_window = states[t-1]
        map_values = state_window[:, MAP_IDX]
        hr_values = state_window[:, HR_IDX]
        pulsat_values = state_window[:, PULSATILITY_IDX]
        slope1 = np.polyfit(x_vals, map_values, 1)[0]
        slope2 = np.polyfit(x_vals, hr_values, 1)[0]
        slope3 = np.polyfit(x_vals, pulsat_values, 1)[0]
        
        if (slope1 < arbitrary_threshold) and (slope2 < arbitrary_threshold) and (slope3 < arbitrary_threshold):
            opportunities += 1
            if actions[t] > actions[t-1]:
                correct_intensifications += 1
    return correct_intensifications / opportunities if opportunities > 0 else 1.0

def compute_map_air_gradient_physician(states, actions):
    """Calculates a physician AIR score for MAP.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken.

    Args:
        states is a 2D array
        actions is a 1D array

    Returns:
        The AIR score for the metric for the length of states, which is a float
    """
    air_value = 0.0
    x_vals = np.arange(FORECAST_HORIZON)
    for t in range(1, len(actions)):
        window_states = states[t-1]
        map_values = window_states[:, MAP_IDX]
        slope = np.polyfit(x_vals, map_values, 1)[0]
        action_change = actions[t] - actions[t-1]
        air_value += max(0, -slope) * max(0, action_change)
    return air_value

def compute_hr_air_gradient_physician(states, actions):
    """Calculates a physician AIR score for HR.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken.

    Args:
        states is a 2D array
        actions is a 1D array

    Returns:
        The AIR score for the metric for the length of states, which is a float
    """
    air_value = 0.0
    x_vals = np.arange(FORECAST_HORIZON)
    for t in range(1, len(actions)):
        window_states = states[t-1]
        hr_values = window_states[:, HR_IDX]
        slope = np.polyfit(x_vals, hr_values, 1)[0]
        action_change = actions[t] - actions[t-1]
        air_value += max(0, -slope) * max(0, action_change)
    return air_value

def compute_pulsat_air_gradient_physician(states, actions):
    """Calculates a physician AIR score for pulsatility.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken.

    Args:
        states is a 2D array
        actions is a 1D array

    Returns:
        The AIR score for the metric for the length of states, which is a float
    """
    air_value = 0.0
    x_vals = np.arange(FORECAST_HORIZON)
    for t in range(1, len(actions)):
        window_states = states[t-1]
        pulsat_values = window_states[:, PULSATILITY_IDX]
        slope = np.polyfit(x_vals, pulsat_values, 1)[0]
        action_change = actions[t] - actions[t-1] 
        air_value += max(0, -slope) * max(0, action_change)
    return air_value



def compute_aggregate_air_gradient_physician(states, actions):
    """Calculates a cumulative aggregate physician AIR score.

    The score is the product of the magnitude of negative gradients 
    within an hour and positive actions taken. The gradient is determined
    as the mean of MAP, HR, and Pulsatility gradients

    Args:
        states is a 2D array with all of the physiological metrics
        actions is a 1D array

    Returns:
        The cumulative aggregate AIR score, which is a float
    """
    air_value = 0.0
    x_vals = np.arange(FORECAST_HORIZON)
    for t in range(1, len(actions)):
        state_window = states[t-1]
        map_values = state_window[:, MAP_IDX]
        hr_values = state_window[:, HR_IDX]
        pulsat_values = state_window[:, PULSATILITY_IDX]
        slope_map = np.polyfit(x_vals, map_values, 1)[0]
        slope_hr = np.polyfit(x_vals, hr_values, 1)[0]
        slope_pulsat = np.polyfit(x_vals, pulsat_values, 1)[0]
        mean_slope = (slope_map + slope_hr + slope_pulsat) / 3.0
        action_change = actions[t] - actions[t-1]
        air_value += max(0, -mean_slope) * max(0, action_change)
    return air_value