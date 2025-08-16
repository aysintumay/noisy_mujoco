#!/usr/bin/env python3
"""
Example script demonstrating how to use the AbiomedRLEnv for reinforcement learning.

This script shows how to:
1. Create the RL environment
2. Train a simple random policy
3. Evaluate the policy
4. Compare with MPC baseline
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import argparse
import json
import pickle
import gym
from datetime import datetime
from stable_baselines3 import DQN, A2C, PPO, SAC
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.logger import configure



from rl_env import AbiomedRLEnvFactory
from mpc import mpc_planning


class RandomPolicy:
    """Simple random policy for baseline comparison."""
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
    
    def get_env(self):
        return self.env
    
    def get_action(self, observation):
        return self.action_space.sample()
    
    def learn(self, total_timesteps: int = 100):

        """Train a policy for a given number of episodes."""
        episode_rewards = []

        for episode in range(total_timesteps):
            obs, info = self.env.reset()
            total_reward = 0
            
            for step in range(self.env.max_steps):
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update policy if it supports learning
                if hasattr(self, 'update'):
                    self.update(obs, action, reward, next_obs, terminated or truncated)
                
                total_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                if hasattr(self, 'epsilon'):
                    print(f"Episode {episode}: Average reward (last 10) = {avg_reward:.3f}, Epsilon = {self.epsilon:.3f}")
                else:
                    print(f"Episode {episode}: Average reward (last 10) = {avg_reward:.3f}")

        return episode_rewards


def evaluate_policy(policy, num_episodes: int = 50) -> Dict[str, float]:
    """Evaluate a trained policy."""
    episode_rewards = []
    episode_lengths = []
    env = policy.get_env()

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(env.max_steps):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards
    }

def evaluate_policy_stable_baselines(policy, num_episodes: int = 50, max_steps: int = 24) -> Dict[str, float]:
    """Evaluate a trained policy using stable baselines."""

    env = policy.get_env()
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
        print(f"Wrapped environment in DummyVecEnv")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = policy.predict(obs)
            obs, reward, dones, info = env.step(action)
            total_reward += reward
            steps += 1
    
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    episode_rewards = np.array(episode_rewards).flatten()

    return {
        "mean_reward": np.mean(episode_rewards).item(),
        "std_reward": np.std(episode_rewards).item(),
        "min_reward": np.min(episode_rewards).item(),
        "max_reward": np.max(episode_rewards).item(),
        "mean_length": np.mean(episode_lengths).item(),
        "episode_rewards": episode_rewards.tolist()
    }
    
    

def run_mpc_baseline(env, num_episodes: int = 50) -> Dict[str, float]:
    """Run MPC as a baseline comparison."""
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        
        # Convert numpy observation back to torch tensor for MPC
        horizon = env.world_model.forecast_horizon
        features = env.world_model.num_features
        initial_state = torch.tensor(obs, dtype=torch.float32).reshape(1, horizon, features)
        
        # Run MPC planning
        actions, outputs = mpc_planning(
            env.world_model,
            initial_state,
            planning_horizon=3,
            n_plans=20,
            n_steps=min(env.max_steps, 10)  # Limit MPC steps for efficiency
        )
        
        # Execute MPC actions
        for i, action in enumerate(actions):
            if i >= env.max_steps:
                break
            
            # Convert MPC action (p-level) to environment action
            if env.action_space_type == "discrete":
                env_action = action - 2  # Convert p-level 2-10 to action 0-8
            else:
                env_action = np.array([action], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(env_action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"MPC Episode {episode}: Reward = {total_reward:.3f}")
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "episode_rewards": episode_rewards
    }



def main():
    parser = argparse.ArgumentParser(description="RL Environment Example")
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--train_episodes", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--policy_type", type=str, default="random", 
                       choices=["random","sac", "dqn", "ppo", "a2c"])
    parser.add_argument("--normalize_rewards", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--mpc", type=bool, default=False)
    parser.add_argument("--results_path", type=str, default="")
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args()
    
    print("Creating RL environment...")
    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        max_steps=args.max_steps,
        action_space_type="discrete",
        reward_type="smooth",
        normalize_rewards=args.normalize_rewards,
        seed=42
    )
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    model_device = env.world_model.device
    print(f"World model device: {model_device}")
    
    policy_kwargs = {
        "batch_size": args.batch_size,
    }
    # Create policy
    if args.policy_type == "random":
        policy = RandomPolicy(env)
        print("Using random policy")
    elif args.policy_type == "dqn":
        policy = DQN("MlpPolicy", env, verbose=1, device=model_device, **policy_kwargs)
        print("Using DQN policy")
    elif args.policy_type == "a2c":
        policy = A2C("MlpPolicy", env, verbose=1, device=model_device)
        print("Using A2C policy")
    elif args.policy_type == "sac":
        env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        max_steps=args.max_steps,
        action_space_type="continuous",
        reward_type="smooth",
        normalize_rewards=args.normalize_rewards,
        seed=42
        )
        policy = SAC("MlpPolicy", env, verbose=1, device=model_device, **policy_kwargs)
        print("Using SAC policy")
    elif args.policy_type == "ppo":
        policy = PPO("MlpPolicy", env, verbose=1, device=model_device, **policy_kwargs)
        print("Using PPO policy")
    
    if args.policy_type != "random":
        print(f"Policy created on device: {policy.device}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    log_results_name = f"{args.results_path}{args.policy_type}_{time_str}"
    if args.policy_type in ["dqn", "a2c", "ppo", "sac"]:
        tmp_path = f"sb3_log/{log_results_name}"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        policy.set_logger(new_logger)
    
    # Train policyπ
    print(f"\nTraining policy for {args.train_episodes} episodes...")
    policy.learn(total_timesteps=args.train_episodes*args.max_steps)
    
    # Save model if specified
    if hasattr(policy, 'save'):
        policy.save(f"models/{log_results_name}")
        print(f"Model saved to models/{log_results_name}")
    
    # Evaluate policy
    print(f"\nEvaluating policy for {args.eval_episodes} episodes...")
    if args.policy_type in ["dqn", "a2c", "ppo", "sac"]:
        eval_results = evaluate_policy_stable_baselines(policy, args.eval_episodes)
    else:
        eval_results = evaluate_policy(policy, args.eval_episodes)
    
    print(f"\nEvaluation Results:")
    print(f"Mean reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"Min reward: {eval_results['min_reward']:.3f}")
    print(f"Max reward: {eval_results['max_reward']:.3f}")
    print(f"Mean episode length: {eval_results['mean_length']:.1f}")
    
    # Run MPC baseline
    if args.mpc:
        print(f"\nRunning MPC baseline for {args.eval_episodes} episodes...")
        mpc_results = run_mpc_baseline(env, args.eval_episodes)
    
        print(f"\nMPC Baseline Results:")
        print(f"Mean reward: {mpc_results['mean_reward']:.3f} ± {mpc_results['std_reward']:.3f}")
        print(f"Min reward: {mpc_results['min_reward']:.3f}")
        print(f"Max reward: {mpc_results['max_reward']:.3f}")
    else:
        mpc_results = {"mean_reward": 0.0, "std_reward": 0.0, "min_reward": 0.0, "max_reward": 0.0}
    
    # Save results
    results = {
        "policy_type": args.policy_type,
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "policy_results": eval_results,
        "mpc_results": mpc_results,
    }
    


    with open(f"results/{log_results_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"results/{log_results_name}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {log_results_name}.json and {log_results_name}.pkl")
    
    env.close()


if __name__ == "__main__":
    main() 