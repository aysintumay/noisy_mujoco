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
    

def train_sac(env, args):
    
    model = SAC("MlpPolicy", env, verbose=1, learning_starts=10000, device = f"cuda:{args.devid}", seed=args.seed)
    model.learn(total_timesteps=args.steps, log_interval=10)
    savedir = os.path.join(args.log_dir, 'expert_models', f"sac_v2_{args.env_name.split('-')[0]}")
    model.save(savedir)


    print(f'SAC trained in {args.env_name} for {args.steps}')
    print(f"SAC training completed and model saved in {savedir}.")
    del model # remove to demonstrate saving and loading

def train_tqc(env, args):
    model = TQC("MlpPolicy", env, verbose=1, device = f"cuda:{args.devid}", seed=args.seed)
    model.learn(total_timesteps=args.steps, log_interval=4)
    savedir = os.path.join(args.log_dir, 'expert_models', f"tqc_{args.env_name.split('-')[0]}")
    model.save(savedir)


    print(f'TQC trained in {args.env_name} for {args.steps}')
    print(f"TQC training completed and model saved in {savedir}.")
    del model # remove to demonstrate saving and loading



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Hopper-v2")
    parser.add_argument("--steps", type=int, default=25e6, help="Total number of steps to train the agent")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to create in the dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="/abiomed/intermediate_data_d4rl")
    parser.add_argument("--devid", type=int, default=3)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    if (args.env_name == "Hopper-v2" or args.env_name == "Walker2d-v2"):
        train_sac(env, args)
    elif args.env_name == "HalfCheetah-v2":
        train_tqc(env, args)
    else: 
        raise ValueError("Unsupported environment for training SAC or TQC.")

   
    env.close()
    print()