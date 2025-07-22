import gymnasium as gym

from stable_baselines3 import SAC
from sb3_contrib import TQC
# Import necessary libraries
import argparse
import os

import sys
import numpy as np
import pickle
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from models.wrappers import (
                        RandomNormalNoisyActions, 
                        RandomNormalNoisyTransitions,
                        RandomNormalNoisyTransitionsActions
                        )

def get_savedir(args):
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
    return logdir


def train_sac(env, args):
    
    model = SAC("MlpPolicy", env, verbose=1, learning_starts=10000, device = f"cuda:{args.devid}", seed=args.seed, tensorboard_log='log')
    model.learn(total_timesteps=args.steps, log_interval=50)
    savedir = get_savedir(args)
    model.save(savedir)

    print(f"SAC training completed with {args.steps} steps and model saved in {savedir}.")
    del model # remove to demonstrate saving and loading

def train_tqc(env, args):
    model = TQC("MlpPolicy", env, verbose=1, learning_starts=10000, device = f"cuda:{args.devid}", seed=args.seed, tensorboard_log='log')
    model.learn(total_timesteps=args.steps, log_interval=4)
    savedir = get_savedir(args)
    model.save(savedir)

    print(f"TQC training completed with {args.steps} steps and model saved in {savedir}.")
    del model # remove to demonstrate saving and loading



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Hopper-v2")
    parser.add_argument("--steps", type=int, default=25e6, help="Total number of steps to train the agent")
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="/abiomed/intermediate_data_d4rl")
    parser.add_argument("--devid", type=int, default=3)

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

    if (args.env_name == "Hopper-v2" or args.env_name == "Walker2d-v2"):
        train_sac(noisy_env, args)
    elif args.env_name == "HalfCheetah-v2":
        train_tqc(noisy_env, args)
    else: 
        raise ValueError("Unsupported environment for training SAC or TQC.")

   
    env.close()
    noisy_env.close()
    print()