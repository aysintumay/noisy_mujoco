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
import os


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wrappers import (
                        RandomNormalNoisyActions, 
                        RandomNormalNoisyTransitions,
                        RandomNormalNoisyTransitionsActions
                        )
from create_dataset import load_policy, load_policy_farama





def evaluate_expert(env, model, args):
    print(env)
    
    obs, _ = env.reset(seed=args.seed)
    num_episodes = 0
    eval_ep_info_buffer = []
    episode_reward, episode_length = 0, 0

    
    while num_episodes <= args.episodes:
        ( action, _ )= model.predict(obs, deterministic=True)
        if args.action:
            action = env.action(action)
        next_obs, reward, terminal, truncated, _ = env.step(action) 
        episode_reward += reward
        episode_length += 1

        obs = next_obs

        if terminal or truncated:
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward, "episode_length": episode_length}
            )

            num_episodes +=1
            episode_reward, episode_length = 0, 0
            obs,_ = env.reset(seed=args.seed)
    df =  {
        "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
        "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
    }

    avg_r = np.mean(df["eval/episode_reward"])
    avg_l = np.mean(df["eval/episode_length"])
    print(f"Average episode reward: {avg_r}, Average episode length: {avg_l}")

    return avg_r, avg_l
    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Hopper-v2")
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--log_dir", type=str, default="/abiomed/intermediate_data_d4rl")
    parser.add_argument("--devid", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate the expert")
    parser.add_argument("--farama", action='store_true', help="Use farama minari expert policy")

    args = parser.parse_args()

    env = gym.make(args.env_name)
    # args.transition = True
    # args.action = True

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

    evaluate_expert(noisy_env,model, args)
    env.close()
    print()

    