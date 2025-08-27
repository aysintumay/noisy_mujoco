import gym
import numpy as np
import torch
from gym import spaces
from typing import Dict, Any, Tuple, Optional
import random

#add path of where this file is located
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import WorldModel
from reward_func import compute_reward_smooth,compute_shaped_reward
import config as config


class AbiomedRLEnv(gym.Env):
    """RL Environment for Abiomed World Model"""
    
    def __init__(
        self,
        world_model: WorldModel,
        max_steps: int = 24,
        action_space_type: str = "discrete",
        reward_type: str = "smooth",
        normalize_rewards: bool = True,
        # reshaped_rwd: bool = False,
        seed: Optional[int] = None,
        gamma1: Optional[float] = 0.0,
        gamma2: Optional[float] = 0.0,
        gamma3: Optional[float] = 0.0,
    ):
        super().__init__()
        
        self.world_model = world_model
        self.max_steps = max_steps
        self.action_space_type = action_space_type
        self.reward_type = reward_type
        self.normalize_rewards = normalize_rewards
        # self.reshaped_rwd = reshaped_rwd
    
        if seed is not None:
            self.seed(seed)

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        
        # Episode tracking
        self.current_step = 0
        self.current_state = None
        self.episode_rewards = []
        self.episode_actions = []
        
        # Action space
        if action_space_type == "discrete":
            self.action_space = spaces.Discrete(9)  # p-levels 2-10
            self.action_mapping = {i: i + 2 for i in range(9)}
        else:
            self.min_action = self.world_model.normalize_pl(torch.tensor([2])).item()
            self.max_action = self.world_model.normalize_pl(torch.tensor([10])).item()
            print(f"Continuous action space. Min action: {self.min_action}, Max action: {self.max_action}")
            self.action_space = spaces.Box(
                low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
            )
        
        # Observation space (all features)
        obs_dim = self.world_model.num_features * self.world_model.forecast_horizon
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(obs_dim,), dtype=np.float32
        )
        
    def _get_next_episode_start(self, idx=None) -> torch.Tensor:

        # pick a random initial state from the data (always in distribution)

        train_length = len(self.world_model.data_train)
        val_length = len(self.world_model.data_val)
        test_length = len(self.world_model.data_test)
        
        init_data_size = train_length + val_length + test_length
        if init_data_size == 0:
            raise ValueError("No data available")
        
        if idx== None:
            init_data_index = random.randint(0, init_data_size - self.max_steps)
        else:
            init_data_index = idx

        if init_data_index < train_length:
            if init_data_index + self.max_steps> train_length:
                full_state = torch.concat([self.world_model.data_train[init_data_index: ][0],\
                                            self.world_model.data_val[: self.max_steps - (train_length - init_data_index)][0]])
            else: 
                full_state = self.world_model.data_train[init_data_index: init_data_index + self.max_steps][0]
            return self.world_model.data_train[init_data_index][0], full_state
        elif init_data_index < train_length + val_length:
            if init_data_index - train_length + self.max_steps > val_length:
                full_state = torch.concat([self.world_model.data_val[init_data_index - train_length: ][0],\
                                            self.world_model.data_test[: self.max_steps - (val_length - (init_data_index - train_length))][0]])
            else:
                full_state = self.world_model.data_val[init_data_index - train_length: init_data_index - train_length + self.max_steps][0]
            return self.world_model.data_val[init_data_index - train_length][0], full_state
        else:
            
            return self.world_model.data_test[init_data_index - train_length - val_length][0], \
                self.world_model.data_test[init_data_index - train_length - val_length: init_data_index - train_length - val_length + self.max_steps][0]     
    
    def _action_to_p_level(self, action) -> int:
        if self.action_space_type == "discrete":
            return self.action_mapping[int(action)]
        else:
            # takes unnormalized action output and returns p-level
            unnormed_action = self.world_model.unnorm_pl(torch.tensor(action))
            return int(np.clip(unnormed_action, 2, 10))
    
    def _compute_reward(self, next_state: torch.Tensor, state: Optional[torch.Tensor] = None, actions:Optional[torch.Tensor] = None) -> float:
        """
        state: torch.Tensor, shape (1 * forecast_horizon,feature_dim), normalized
        actions: list, shape (1, 2) or None, unnormalized
        """
        next_state_reshaped = next_state.cpu().unsqueeze(0)
        next_state_reshaped_unnorm = self.world_model.unnorm_output(next_state_reshaped)
          
        reward = compute_reward_smooth(next_state_reshaped_unnorm).item()
        
        if self.normalize_rewards:
            # OLD: mean and std of original + low_p datasets are -8.93 and 4.41
            mean = -1.7018
            std = 2.6621
            reward = (reward - mean) / std
            reward = np.clip(reward, -2.0, a_max = None) # ~4% is clipped from below

        if (actions is not None) and (state is not None):
            #self.current_state
            state_reshaped = state.cpu().unsqueeze(0) 
            state_reshaped_unnorm = self.world_model.unnorm_output(state_reshaped) 
            add_rwd = compute_shaped_reward(state_reshaped_unnorm, actions, self.gamma1, self.gamma2, self.gamma3)
            reward += add_rwd
        return reward
    
    def _get_observation(self, state: torch.Tensor) -> np.ndarray:
        return state.cpu().numpy().reshape(-1).astype(np.float32)

    def _get_all_observations(self, state: torch.Tensor) -> np.ndarray:
        return state.cpu().numpy().reshape(self.max_steps, -1).astype(np.float32)
    
    def reset(self, idx: Optional[int] = None, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        
        if options is not None:
            print(f"Resetting with options: {options}")
            self.current_state = options["state"]
        else:
            self.current_state, all_states = self._get_next_episode_start(idx)
        
        self.current_step = 0
        
        self.episode_rewards = []
        self.episode_actions = []
        
        observation = self._get_observation(self.current_state)
        all_observations = self._get_all_observations(all_states)
        
        info = {"all_states":all_observations, "episode": {"r": 0.0, "l": 0, "t": 0.0}}
        
        return observation, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        p_level = self._action_to_p_level(action)
        
        self.episode_actions.append(p_level)
        
        with torch.no_grad():
            next_state = self.world_model.step(self.current_state.unsqueeze(0), p_level).squeeze(0)
        
        self.current_state = next_state
        
        reward = self._compute_reward(next_state)
        self.episode_rewards.append(reward)
        
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        observation = self._get_observation(next_state)
        #print("current step", self.current_step)
        info = {
            "p_level": p_level,
            "step": self.current_step,
            "episode": {
                "r": sum(self.episode_rewards),
                "l": self.current_step
            } if terminated or truncated else None
        }
        
        return observation, reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        if not self.episode_rewards:
            return {}
        
        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "episode_length": len(self.episode_rewards),
            "actions_taken": self.episode_actions.copy()
        }
    
    def render(self, mode: str = "human"):
        pass
    
    def close(self):
        pass
    
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
    
   
class AbiomedRLEnvNoisy(AbiomedRLEnv):
    """RL Environment for Abiomed World Model with noise"""
    
    def __init__(self, *args, noise_rate: float = 0.0, 
                              noise_scale: float = 0.00,
                              **kwargs):

        super().__init__(*args, **kwargs)
        self.noise_rate = noise_rate
        self.noise_scale = noise_scale
        print(f"Noise rate: {self.noise_rate}, Noise scale: {self.noise_scale}")
    
    def add_noise(self, observation: np.ndarray) -> np.ndarray:
        """Add noise to the observation"""
        # do not add noise to the p-levels
        noise = np.random.normal(0, self.noise_scale, observation.shape)
        # p level is last column of the observation
        num_features = self.world_model.num_features
        for i in range(self.world_model.forecast_horizon):
            noise[i * num_features + num_features - 1] = 0
        observation = observation + noise
        return observation

    def reset(self, *args, **kwargs):
        observation, info = super().reset(*args, **kwargs)
        if np.random.uniform(0, 1) < self.noise_rate:
            observation = self.add_noise(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if np.random.uniform(0, 1) < self.noise_rate:
            observation = self.add_noise(observation)
        return observation, reward, terminated, truncated, info
    


class AbiomedRLEnvFactory:
    """Factory for creating AbiomedRLEnv instances"""
    
    @staticmethod
    def create_env(
        model_name: str = "10min_1hr_window",
        model_path: str = None,
        data_path: str = None,
        max_steps: int = 50,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
        gamma3: float = 0.0,
        action_space_type: str = "discrete",
        reward_type: str = "smooth",
        normalize_rewards: bool = True,
        noise_rate: float = 0.0,
        noise_scale: float = 0.00,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ) -> AbiomedRLEnv:
        
        print(config)
        if model_name not in config.model_configs:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        model_kwargs = config.model_configs[model_name]
        model_kwargs['device'] = torch.device(device) if device else torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        world_model = WorldModel(**model_kwargs)
        
        if model_path is None:
            model_path = f"/abiomed/downsampled/models/{model_name}_model.pth"

        world_model.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        if data_path is None:
            data_path = f"/abiomed/downsampled/{model_name}.pkl"
        
        world_model.load_data(data_path)
        print(f"Data loaded from {data_path}")

        if noise_rate > 0:
            env = AbiomedRLEnvNoisy(
                world_model=world_model,
                max_steps=max_steps,
                action_space_type=action_space_type,
                reward_type=reward_type,
                normalize_rewards=normalize_rewards,
                seed=seed,
                noise_rate=noise_rate,
                noise_scale=noise_scale
            )
        else:
            env = AbiomedRLEnv(
                world_model=world_model,
                max_steps=max_steps,
                action_space_type=action_space_type,
                reward_type=reward_type,
                normalize_rewards=normalize_rewards,
                seed=seed
            )
        
        return env


if __name__ == "__main__":
    # Test the environment
    try:
        env = AbiomedRLEnvFactory.create_env(
            model_name="10min_1hr_window",
            max_steps=24,
            action_space_type="discrete",
            reward_type="smooth",
            normalize_rewards=True,
            seed=42
        )
        
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test episode
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        stats = env.get_episode_stats()
        print(f"Test episode: Total reward = {total_reward:.3f}, "
              f"Steps = {stats['episode_length']}")
        
        env.close()
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        print("This is expected if model/data files are not available.") 