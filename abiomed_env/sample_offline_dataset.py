#!/usr/bin/env python3
"""
Script to sample a dataset from AbiomedRLEnv using a trained SAC policy in D4RL format.

This script loads a trained SAC policy and uses it to generate trajectories 
in the AbiomedRLEnv, then formats the data according to D4RL specifications.

Usage:
    python sample_offline_dataset.py --model_path models/sac_20250827_0837.zip --num_episodes 1000
"""

import numpy as np
import pickle
import argparse
import os
from typing import Dict, List, Any
from stable_baselines3 import SAC
from rl_env import AbiomedRLEnvFactory
import torch
from tqdm import tqdm


def sample_trajectories_from_policy(
    policy: SAC,
    env, 
    num_episodes: int,
    deterministic: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Sample trajectories from a trained SAC policy.
    
    Args:
        policy: Trained SAC policy
        env: AbiomedRLEnv environment
        num_episodes: Number of episodes to sample
        deterministic: Whether to use deterministic policy actions
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing trajectory data in D4RL format
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    next_observations = []
    infos = []
    
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_obs = [obs]
        episode_actions = []
        episode_rewards = []
        episode_terminals = []
        episode_timeouts = []
        episode_next_obs = []
        episode_infos = [info]
        
        done = False
        step = 0
        
        while not done:
            # Get action from policy
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            unnormed_action = env.world_model.unnorm_pl(torch.tensor(action))
            action = int(np.clip(unnormed_action, 2, 10))
            
            # Store data
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_terminals.append(terminated)
            episode_timeouts.append(truncated)
            episode_next_obs.append(next_obs)
            episode_infos.append(info)
            
            # Update observation
            obs = next_obs
            episode_obs.append(obs)
            step += 1
        
        # Convert to numpy arrays and add to dataset
        episode_obs = np.array(episode_obs[:-1])  # Remove last observation
        episode_next_obs = np.array(episode_next_obs)
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)
        episode_terminals = np.array(episode_terminals)
        episode_timeouts = np.array(episode_timeouts)
        
        # Add episode data to full dataset
        observations.append(episode_obs)
        next_observations.append(episode_next_obs)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        terminals.append(episode_terminals)
        timeouts.append(episode_timeouts)
        infos.extend(episode_infos[:-1])  # Remove last info
        
        if verbose and episode % 50 == 0:
            print(f"Episode {episode}/{num_episodes}: {len(episode_rewards)} steps, "
                  f"total reward: {np.sum(episode_rewards):.3f}")
    
    # Concatenate all episodes
    dataset = {
        'observations': np.concatenate(observations, axis=0),
        'next_observations': np.concatenate(next_observations, axis=0),
        'actions': np.concatenate(actions, axis=0),
        'rewards': np.concatenate(rewards, axis=0),
        'terminals': np.concatenate(terminals, axis=0),
        'timeouts': np.concatenate(timeouts, axis=0),
        'infos': infos
    }
    
    if verbose:
        print(f"\nDataset created:")
        print(f"  Total transitions: {len(dataset['observations'])}")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Observation shape: {dataset['observations'].shape}")
        print(f"  Action shape: {dataset['actions'].shape}")
        print(f"  Average episode length: {len(dataset['observations']) / num_episodes:.1f}")
        print(f"  Average reward per transition: {np.mean(dataset['rewards']):.3f}")
        print(f"  Reward std: {np.std(dataset['rewards']):.3f}")
    
    return dataset


def save_dataset(dataset: Dict[str, np.ndarray], save_path: str, metadata: Dict[str, Any] = None):
    """
    Save the dataset in D4RL compatible format.
    
    Args:
        dataset: Dictionary containing trajectory data
        save_path: Path to save the dataset
        metadata: Optional metadata to include
    """
    # Add metadata
    if metadata is not None:
        dataset['metadata'] = metadata
    
    # Save as pickle file (D4RL compatible)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to: {save_path}")
    
    # Also save as numpy archive for easier access
    npz_path = save_path.replace('.pkl', '.npz')
    dataset_without_infos = {k: v for k, v in dataset.items() if k != 'infos' and k != 'metadata'}
    np.savez_compressed(npz_path, **dataset_without_infos)
    print(f"Dataset also saved as: {npz_path}")


def validate_dataset(dataset: Dict[str, np.ndarray]):
    """
    Validate that the dataset follows D4RL format requirements.
    
    Args:
        dataset: Dictionary containing trajectory data
    """
    required_keys = ['observations', 'actions', 'rewards', 'terminals', 'timeouts']
    optional_keys = ['next_observations', 'infos', 'metadata']
    
    # Check required keys
    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Missing required key: {key}")
    
    # Check array shapes are consistent
    n_transitions = len(dataset['observations'])
    for key in ['actions', 'rewards', 'terminals', 'timeouts']:
        if len(dataset[key]) != n_transitions:
            raise ValueError(f"Inconsistent length for {key}: {len(dataset[key])} vs {n_transitions}")
    
    # Check next_observations if present
    if 'next_observations' in dataset:
        if len(dataset['next_observations']) != n_transitions:
            raise ValueError(f"Inconsistent length for next_observations: {len(dataset['next_observations'])} vs {n_transitions}")
    
    print("Dataset validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sample D4RL dataset from trained SAC policy")
    parser.add_argument("--model_path", type=str, default="models/sac_20250827_0837.zip",
                       help="Path to trained SAC model (.zip file)")
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data",
                       help="Model name for environment configuration")
    parser.add_argument("--num_episodes", type=int, default=5000,
                       help="Number of episodes to sample")
    parser.add_argument("--max_steps", type=int, default=6,
                       help="Maximum steps per episode")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy actions")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save the dataset (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default='cuda:1',
                       help="Device to run on (cuda:0, cuda:1, cpu)")
    parser.add_argument("--noise_rate", type=float, default=0.0,
                       help="Noise rate for adding noise to the observations")
    parser.add_argument("--noise_scale", type=float, default=0.00,
                       help="Scale for adding noise to the observations")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device is None:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        max_steps=args.max_steps,
        action_space_type="continuous",  # SAC requires continuous action space
        reward_type="smooth",
        normalize_rewards=True,
        seed=args.seed,
        device=device,
        noise_rate=args.noise_rate,
        noise_scale=args.noise_scale
    )
    
    print(f"Environment created!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Load trained SAC policy
    print(f"Loading SAC policy from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    policy = SAC.load(args.model_path, env=env, device=device)
    print(f"Policy loaded successfully on device: {policy.device}")
    
    # Sample trajectories
    print(f"\nSampling {args.num_episodes} episodes...")
    dataset = sample_trajectories_from_policy(
        policy=policy,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        verbose=True
    )
    
    # Validate dataset
    validate_dataset(dataset)
    
    # Generate save path if not provided
    if args.save_path is None:
        model_basename = os.path.basename(args.model_path).replace('.zip', '')
        deterministic_str = "_deterministic" if args.deterministic else "_stochastic"
        noise_str = f"_noise{args.noise_rate:.2f}_scale{args.noise_scale:.2f}" if args.noise_rate > 0 else ""
        save_path = f"/abiomed/offline_datasets/SAC_exp_{args.num_episodes}eps{deterministic_str}{noise_str}.pkl"
    else:
        save_path = args.save_path
    
    # Create metadata
    if args.noise_rate > 0:
        env_type = "AbiomedRLEnvNoisy"
    else:
        env_type = "AbiomedRLEnv"

    metadata = {
        'model_path': args.model_path,
        'model_name': args.model_name,
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'deterministic': args.deterministic,
        'seed': args.seed,
        'environment': env_type,
        'action_space_type': 'continuous',
        'policy_type': 'SAC',
        'noise_rate': args.noise_rate,
        'noise_scale': args.noise_scale
    }
    
    # Save dataset
    save_dataset(dataset, save_path, metadata)
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Total transitions: {len(dataset['observations'])}")
    print(f"  Average episode length: {len(dataset['observations']) / args.num_episodes:.1f}")
    print(f"  Reward statistics:")
    print(f"    Mean: {np.mean(dataset['rewards']):.4f}")
    print(f"    Std: {np.std(dataset['rewards']):.4f}")
    print(f"    Min: {np.min(dataset['rewards']):.4f}")
    print(f"    Max: {np.max(dataset['rewards']):.4f}")
    print(f"  Terminal rate: {np.mean(dataset['terminals']):.4f}")
    print(f"  Timeout rate: {np.mean(dataset['timeouts']):.4f}")
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()