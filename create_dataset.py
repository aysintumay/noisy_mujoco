import gymnasium as gym
from huggingface_sb3 import load_from_hub
from huggingface_hub import hf_hub_download

from stable_baselines3 import SAC
from sb3_contrib import TQC
# Import necessary libraries
import argparse
import os
import sys
import numpy as np
import pickle
import tqdm
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.base_class import BaseAlgorithm
import zipfile
import os
import tempfile
import torch
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wrappers import (
                        RandomNormalNoisyActions, 
                        RandomNormalNoisyTransitions,
                        RandomNormalNoisyTransitionsActions
                        )

def load_noisy_sac_policy(args):
    if args.action and not args.transition:
        logdir = os.path.join(args.log_dir, 'sac_expert', f"sac_v2_{args.env_name.split('-')[0]}_action_noisy_{args.scale_action}_{args.noise_rate_action}.pkl")   
        print(f"Training with noisy actions in {logdir}")
    elif args.transition and not args.action:
        logdir = os.path.join(args.log_dir, 'sac_expert', f"sac_v2_{args.env_name.split('-')[0]}_obs_noisy_{args.scale_transition}_{args.noise_rate_transition}.pkl") 
        print(f"Training with noisy transitions in {logdir}")
    elif args.transition and args.action:
        # noisy_env = RandomNormalNoisyTransitionsActions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
        logdir = os.path.join(args.log_dir, 'expert_models', f"sac_v2_{args.env_name.split('-')[0]}_action_obs_noisy_{args.scale_action}_{args.noise_rate_action}_{args.scale_transition}_{args.noise_rate_transition}.pkl") 
        print(f"Training with noisy actions and transitions in {logdir}")
    else:
        logdir = os.path.join(args.log_dir, 'expert_models', f"sac_v2_{args.env_name.split('-')[0]}")
        print(f'Training in {logdir}!')
    
    model = SAC.load(logdir, device = f"cuda:{args.devid}")
    return model
    
    


  
def load_policy_farama(env, args):
    if (args.env_name.split('-')[0] == "Hopper") or (args.env_name.split('-')[0] == "Walker2d"):
        algo = "sac"
        
    elif args.env_name.split('-')[0] == "HalfCheetah":
        algo = "tqc"
    else:
        raise ValueError("Unsupported environment for noisy datasets creation.")
    
    model_checkpoint = load_from_hub(
        repo_id=f"farama-minari/{args.env_name.split('-')[0]}-v5-{algo.upper()}-expert",
        filename=f"{args.env_name.split('-')[0].lower()}-v5-{algo.upper()}-expert.zip",
    )
    if (args.env_name.split('-')[0] == "Hopper") or (args.env_name.split('-')[0] == "Walker2d"):
        model = SAC.load(model_checkpoint, env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space}, device = f"cuda:{args.devid}")

    elif args.env_name.split('-')[0] == "HalfCheetah":
        model = TQC.load(model_checkpoint, device = f"cuda:{args.devid}")

    else:
        raise ValueError("Unsupported environment for noisy datasets creation.")
    return model


def load_policy(args):
    
  
    if (args.env_name.split('-')[0] == "Hopper") or (args.env_name.split('-')[0] == "Walker2d"):
         model = SAC.load(os.path.join(args.log_dir, 'expert_models', f"sac_v2_{args.env_name.split('-')[0]}"), device = f"cuda:{args.devid}")

    elif args.env_name.split('-')[0] == "HalfCheetah":
        model = TQC.load(os.path.join(args.log_dir, 'expert_models', f"tqc_{args.env_name.split('-')[0]}"), device = f"cuda:{args.devid}")

    else:
        raise ValueError("Unsupported environment for noisy datasets creation.")
    return model


def main(noisy_env, model, args):

    if args.action and not args.transition:
        # noisy_env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
        if args.farama:
            logdir = os.path.join(args.log_dir, 'farama_sac_expert', f"{args.env_name}_action_noisy_{args.scale_action}_{args.noise_rate_action}.pkl")
        else:
            logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_action_noisy_{args.scale_action}_{args.noise_rate_action}.pkl")   
        print(f"Creating dataset with noisy actions in {logdir}")
    elif args.transition and not args.action:
        # noisy_env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
        if args.farama:
            logdir = os.path.join(args.log_dir, 'farama_sac_expert', f"{args.env_name}_obs_noisy_{args.scale_transition}_{args.noise_rate_transition}.pkl")
        else:
            logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_obs_noisy_{args.scale_transition}_{args.noise_rate_transition}.pkl") 
        print(f"Creating dataset with noisy transitions in {logdir}")
    elif args.transition and args.action:
        # noisy_env = RandomNormalNoisyTransitionsActions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)

        if args.farama:
            logdir = os.path.join(args.log_dir, 'farama_sac_expert', f"{args.env_name}_action_obs_noisy_{args.scale_action}_{args.noise_rate_action}_{args.scale_transition}_{args.noise_rate_transition}.pkl")
        else:
            logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_action_obs_noisy_{args.scale_action}_{args.noise_rate_action}_{args.scale_transition}_{args.noise_rate_transition}.pkl") 
        
        print(f"Creating dataset with noisy actions and transitions in {logdir}")
    else:
        
        if args.farama:
            logdir = os.path.join(args.log_dir, 'farama_sac_expert', f"{args.env_name}_expert_{args.num_samples}.pkl")
        else:
            logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_expert_{args.num_samples}.pkl")
        print(f'Expert dataset is being created in {logdir}!')
        
    
    # observation, info = env.reset(seed = args.seed)
    seed = args.seed
    noisy_data = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}
    # num_samples = 0
    # steps = 0
    truncated = True
    terminated = True
    seed = 123
    for num_samples in  tqdm.tqdm(range(args.dataset_size)):
        
        if terminated or truncated:
            observation, info = noisy_env.reset(seed=seed)
            num_samples += 1
            seed += 1
            if (args.dataset_size - num_samples) < env.spec.max_episode_steps:  # trim trailing non-full episodes
                print(f'endded after {num_samples} steps')
                break

        action, _states = model.predict(observation) 
        if args.action: #add noise here
            action = noisy_env.action(action)

        next_observation, reward, terminated, truncated, info = noisy_env.step(action) #noise added by the environment dynamics

        noisy_data['observations'].append(observation)
        noisy_data['actions'].append(action)
        noisy_data['rewards'].append([reward])
        noisy_data['next_observations'].append(next_observation)
        noisy_data['terminals'].append([terminated])

        observation = next_observation
        # steps +=1
        

        
    noisy_data = {
                key: np.concatenate(noisy_data[key], axis=0)
                for key in noisy_data.keys()
            }
   
    noisy_data['next_observations'] = noisy_data['next_observations'].reshape(-1, next_observation.shape[0])
    noisy_data['observations'] = noisy_data['observations'].reshape(-1, observation.shape[0])
    noisy_data['actions'] = noisy_data['actions'].reshape(-1, action.shape[0])


    with open(logdir, "wb") as f:
        pickle.dump(noisy_data, f) 
    print(f"Dataset created with {args.num_samples} samples and saved to {logdir}")






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Hopper-v2")
    parser.add_argument("--dataset_size", type=int, default=1000000, help="Number of samples to create in the dataset")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of episodes to create in the dataset")
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log_dir", type=str, default="/abiomed/intermediate_data_d4rl")
    parser.add_argument("--devid", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate the expert")
    parser.add_argument("--farama", action='store_true', help="Use farama minari expert policy")

    args = parser.parse_args()

    env = gym.make(args.env_name)

    if args.action and not args.transition:
        print("Environment with noisy actions")
        noisy_env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate_action, loc = args.loc, scale = args.scale_action)
    elif args.transition and not args.action:
        print("Environment with noisy transitions")
        noisy_env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate_transition, loc = args.loc, scale = args.scale_transition)
    elif args.transition and args.action:
        print("Environment with noisy actions and transitions")
        noisy_env = RandomNormalNoisyTransitionsActions(env=env, noise_rate_action=args.noise_rate_action, loc = args.loc, scale_action = args.scale_action,\
                                                         noise_rate_transition=args.noise_rate_transition, scale_transition = args.scale_transition)
    else:
        print("Environment without noise")
        noisy_env = env

    
    if args.farama:
        model = load_policy_farama(env, args)
    else:
        model = load_policy(args)
    main(noisy_env, model,args)
    # evaluate_expert(noisy_env,model, args)
    env.close()
    print()

    