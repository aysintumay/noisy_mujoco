
import numpy as np
import gymnasium as gym
import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

'''
This module is copied and modified from the noisyenv package [1].

[1] R. Khraishi and R. Okhrati, "Simple Noisy Environment Augmentation for Reinforcement Learning," arXiv preprint arXiv:2305.02882, 2023.
'''

class RandomNormalNoisyActions(gym.ActionWrapper):
    """Adds random Normal noise to the observations of the environment.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomNormalNoisyObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomNormalNoisyActions(env, noise_rate=0.1, loc=0.0, scale=0.1)
    """

    def __init__(self, env, noise_rate=0.01, loc=0.0, scale=0.01):
        """Initializes the :class:`RandomNormalNoisyObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the observation each step.
                Defaults to 0.01.
            loc (float, optional): Mean ("centre") of the noise distribution.
                Defaults to 0.0.
            scale (float, optional): Standard deviation (spread or "width") of the noise distribution.
                Must be non-negative. Defaults to 0.01.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.added_noise = True
        self.loc = loc
        self.scale = scale
        
    def seed(self,seed):
        """Set the seed for the random number generator."""
        np.random.seed(seed)
        self.seed = seed 

    def action(self, action):
        """Modify the action by adding or multiplying noise with some probability."""
        if np.random.rand() <= self.noise_rate:
            noise = np.random.normal(loc=self.loc, scale=self.scale, size=action.shape)
            if self.added_noise:
                action = action + noise
            else:
                action = action * noise
            # Clip to ensure action remains valid
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def get_normalized_score(self, reward, cost=0):
        """Get the normalized score for the observation."""
        if hasattr(self.env, "get_normalized_score"):
            return self.env.get_normalized_score(reward, cost)
        else:
            return reward, cost


class RandomNormalNoisyTransitions(gym.ObservationWrapper):
    """Adds random Normal noise to the observations of the environment.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomNormalNoisyObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomNormalNoisyTransitions(env, noise_rate=0.1, loc=0.0, scale=0.1)
    """

    def __init__(self, env, noise_rate=0.01, loc=0.0, scale=0.01):
        """Initializes the :class:`RandomNormalNoisyObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the observation each step.
                Defaults to 0.01.
            loc (float, optional): Mean ("centre") of the noise distribution.
                Defaults to 0.0.
            scale (float, optional): Standard deviation (spread or "width") of the noise distribution.
                Must be non-negative. Defaults to 0.01.
        """
        super().__init__(env)
        self.noise_rate = noise_rate
        self.added_noise = True
        self.loc = loc
        self.scale = scale

    def seed(self,seed):
        """Set the seed for the random number generator."""
        np.random.seed(seed)
        self.seed = seed

    def observation(self, observation):
        """Modify the action by adding or multiplying noise with some probability."""
        if np.random.rand() <= self.noise_rate:
            noise = np.random.normal(loc=self.loc, scale=self.scale, size=observation.shape)
            if self.added_noise:
                observation = observation + noise
            else:
                observation = observation * noise
            # Clip to ensure action remains valid
            observation = np.clip(observation, self.env.observation_space.low, self.env.observation_space.high)
        return observation

    def get_normalized_score(self, reward, cost=0):
        """Get the normalized score for the observation."""
        if hasattr(self.env, "get_normalized_score"):
            return self.env.get_normalized_score(reward, cost)
        else:
            return reward, cost



class RandomNormalNoisyTransitionsActions(gym.Wrapper):
    """Adds random Normal noise to the observations of the environment.

    Example:
        >>> import gymnasium as gym
        >>> from noisyenv.wrappers import RandomNormalNoisyObservation
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = RandomNormalNoisyTransitionsActions(env, noise_rate_action=0.01, loc=0.0, scale_action=0.01, noise_rate_transition=0.01, scale_transition=0.01)
    """

    def __init__(self, env, noise_rate_action=0.01, loc=0.0, scale_action=0.01, noise_rate_transition=0.01, scale_transition=0.01):
        """Initializes the :class:`RandomNormalNoisyObservation` wrapper.

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise_rate (float, optional): Probability of adding noise to the observation each step.
                Defaults to 0.01.
            loc (float, optional): Mean ("centre") of the noise distribution.
                Defaults to 0.0.
            scale (float, optional): Standard deviation (spread or "width") of the noise distribution.
                Must be non-negative. Defaults to 0.01.
        """
        super().__init__(env)
        self.noise_rate_action_ = noise_rate_action
        self.noise_rate_transition_ = noise_rate_transition
        self.added_noise = True
        self.loc = loc
        self.scale_action_ = scale_action
        self.scale_transition_ = scale_transition

    def seed(self,seed):
        """Set the seed for the random number generator."""
        np.random.seed(seed)
        self.seed = seed
        
    def reset(self, **kwargs):
        """Override the reset method to apply noise to the initial observation."""
        observation, info = self.env.reset(**kwargs)
        # Apply noise to the initial observation
        observation = self.observation(observation)
        return observation, info

    def step(self, action):
        """Override the step method to apply noise to the action and transition."""
        # Apply noise to the action before sending it to the environment
        action = self.action(action)
        
        # Step through the environment
        next_observation, reward, terminated, truncated, info = self.env.step(action)

        # Apply noise to the observation from the environment
        next_observation = self.observation(next_observation)

        return next_observation, reward, terminated, truncated, info

    def observation(self, observation):
        """Modify the action by adding or multiplying noise with some probability."""
        if np.random.rand() <= self.noise_rate_transition_:
            noise_obs = np.random.normal(loc=self.loc, scale=self.scale_transition_, size=observation.shape)

            if self.added_noise:
                observation = observation + noise_obs
            else:
                observation = observation * noise_obs
            # Clip to ensure action remains valid
            observation = np.clip(observation, self.env.observation_space.low, self.env.observation_space.high)
        return observation

        
    def action(self, action):
        """Modify the action by adding or multiplying noise with some probability."""
        if np.random.rand() <= self.noise_rate_action_:
            noise_act = np.random.normal(loc=self.loc, scale=self.scale_action_, size=action.shape)

            if self.added_noise:
                action = action + noise_act
            else:
                action = action * noise_act
            # Clip to ensure action remains valid
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action
    
    def get_normalized_score(self, reward, cost=0):
        """Get the normalized score for the observation."""
        if hasattr(self.env, "get_normalized_score"):
            return self.env.get_normalized_score(reward, cost)
        else:
            return reward, cost