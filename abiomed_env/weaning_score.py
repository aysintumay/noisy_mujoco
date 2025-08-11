import numpy as np

def weaning_score(pl, reward):
    """Calculates the weaning score based on patient level and reward at each episode.

    Args:
        pl (list):  list of p-levels in an episode, a value between 2 and 10.
        reward (list): list of rewards at each state in an episode, a value between -1 and 1.

    Returns:
        float: The weaning score, sums a binary score (0,1) at each state in an episode conditioned on a threshold on reward.
    """
    score = 0.0
    denom = 0.0
    for i in range(len(pl)-1):
        if reward[i] > 0.0:
            denom += 1.0
            if (pl[i+1] - pl[i]) == 1:
                score += 1.0
            
        
    return score / denom if denom != 0 else 0.0
    