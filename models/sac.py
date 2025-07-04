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

from wrappers import (
                        RandomNormalNoisyActions, 
                        RandomNormalNoisyTransitions,
                        RandomNormalNoisyTransitionsActions
                        )



def create_dataset(env, args):
    model = SAC.load(os.path.join(args.log_dir, 'expert_models', f"sac_{args.env_name.split('-')[0]}"))
    dataset = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}

    obs, info = env.reset()
    for _ in range(args.num_samples):
        action, _states = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        dataset['observations'].append(obs)
        dataset['actions'].append(action)
        dataset['rewards'].append([reward])
        dataset['next_observations'].append(next_obs)
        dataset['terminals'].append([terminated] or [truncated])
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()
        
    # Convert lists to numpy arrays
    for key in dataset:
        dataset[key] = np.concatenate(dataset[key], axis=0)
        # noisy_data[key].reshape(-1, sample_dict[key].shape[0])
    dataset['next_observations'] = dataset['next_observations'].reshape(-1, next_obs.shape[0])
    dataset['observations'] = dataset['observations'].reshape(-1, obs.shape[0])
    dataset['actions'] = dataset['actions'].reshape(-1, action.shape[0])

    #save in logdir
    log_dir = os.path.join(args.log_dir, 'sac_expert')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, f"{args.env_name}_e.pkl"), "wb") as f:
        pickle.dump(dataset, f)    
    print(f"Dataset created with {args.num_samples} samples and saved to {log_dir}/{args.env_name}_e.pkl")


# def create_noisy_action_dataset(env, args, noise_rate=0.01, loc=0.0, scale=0.01):

#     env_act = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
#     model = SAC.load(os.path.join(args.log_dir, 'expert_models', f"sac_{args.env_name.split('-')[0]}"))
#     observation, info = env_act.reset(seed=333)
#     noisy_data = {'observation': [], 'action': [], 'reward': [], 'next_observation': [], 'terminal': []}
#     for _ in range(args.num_samples):

#         #when SAC comes, sample from the policy
#         action, _states = model.predict(observation, deterministic=True)
#         next_observation, reward, terminated, truncated, info = env_act.step(action)

#         noisy_data['observation'].append(observation)
#         noisy_data['action'].append(action)
#         noisy_data['reward'].append([reward])
#         noisy_data['next_observation'].append(next_observation)
#         noisy_data['terminal'].append([terminated])

#         if terminated or truncated:
#             observation, info = env_act.reset()
#             # break #get one full episode only
        
#     noisy_data = {
#                 key: np.concatenate(noisy_data[key], axis=0)
#                 for key in noisy_data.keys()
#             }
#     noisy_data['next_observation'] = noisy_data['next_observation'].reshape(-1, next_observation.shape[0])

#     noisy_data['observation'] = noisy_data['observation'].reshape(-1, observation.shape[0])

#     noisy_data['action'] = noisy_data['action'].reshape(-1, action.shape[0])

#     log_dir = os.path.join(args.log_dir, 'sac_expert') 
#     with open(os.path.join(log_dir, f"{args.env_name}_action_noisy.pkl"), "wb") as f:
#         pickle.dump(noisy_data, f)    
#     print(f"Dataset created with {args.num_samples} samples and saved to {log_dir}/{args.env_name}_action_noisy.pkl")



# def create_noisy_transition_dataset(env, args, noise_rate=0.01, loc=0.0, scale=0.01):

#     env_obs = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
#     model = SAC.load(os.path.join(args.log_dir, 'expert_models', f"sac_{args.env_name.split('-')[0]}"))
#     observation, info = env.reset(seed=333)
#     noisy_data = {'observation': [], 'action': [], 'reward': [], 'next_observation': [], 'terminal': []}
#     for _ in range(args.num_samples):
#         action, _states = model.predict(observation, deterministic=True)
#         next_observation, reward, terminated, truncated, info = env_obs.step(action)

#         noisy_data['observation'].append(observation)
#         noisy_data['action'].append(action)
#         noisy_data['reward'].append([reward])
#         noisy_data['next_observation'].append(next_observation)
#         noisy_data['terminal'].append([terminated])

#         observation = next_observation
#         if terminated or truncated:
#             observation, info = env_obs.reset()
#             # break #get one full episode only
        
#     noisy_data = {
#                 key: np.concatenate(noisy_data[key], axis=0)
#                 for key in noisy_data.keys()
#             }
   
#     noisy_data['next_observation'] = noisy_data['next_observation'].reshape(-1, next_observation.shape[0])

#     noisy_data['observation'] = noisy_data['observation'].reshape(-1, observation.shape[0])

#     noisy_data['action'] = noisy_data['action'].reshape(-1, action.shape[0])

#     log_dir = os.path.join(args.log_dir, 'sac_expert') 
#     with open(os.path.join(log_dir, f"{args.env_name}_obs_noisy.pkl"), "wb") as f:
#         pickle.dump(noisy_data, f)    
#     print(f"Dataset created with {args.num_samples} samples and saved to {log_dir}/{args.env_name}_obs_noisy.pkl")




def create_noisy_datasets(env, args):

    if args.action and not args.transition:
        noisy_env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
        logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_action_noisy.pkl") 
        print(f"Creating dataset with noisy actions in {logdir}")
    elif args.transition and not args.action:
        noisy_env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
        logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_obs_noisy.pkl") 
        print(f"Creating dataset with noisy transitions in {logdir}")
    elif args.transition and args.action:
        noisy_env = RandomNormalNoisyTransitionsActions(env=env, noise_rate=args.noise_rate, loc = args.loc, scale = args.scale)
        logdir = os.path.join(args.log_dir, 'sac_expert', f"{args.env_name}_action_obs_noisy.pkl") 
        print(f"Creating dataset with noisy actions and transitions in {logdir}")
    else:
        raise ValueError("Please specify either --action or --transition or both.")
    
    if args.env_name == "Hopper-v2":
        model = SAC.load(os.path.join(args.log_dir, 'expert_models', f"sac_{args.env_name.split('-')[0]}"))
    elif args.env_name == "HalfCheetah-v2":
        model = TQC.load(os.path.join(args.log_dir, 'expert_models', f"tqc_{args.env_name.split('-')[0]}"))
    else:
        raise ValueError("Unsupported environment for noisy datasets creation.")
    
    observation, info = env.reset(seed=333)
    noisy_data = {'observation': [], 'action': [], 'reward': [], 'next_observation': [], 'terminal': []}
    for _ in range(args.num_samples):
        action, _states = model.predict(observation, deterministic=True)
        next_observation, reward, terminated, truncated, info = noisy_env.step(action)

        noisy_data['observation'].append(observation)
        noisy_data['action'].append(action)
        noisy_data['reward'].append([reward])
        noisy_data['next_observation'].append(next_observation)
        noisy_data['terminal'].append([terminated])

        observation = next_observation
        if terminated or truncated:
            observation, info = noisy_env.reset()
            # break #get one full episode only
        
    noisy_data = {
                key: np.concatenate(noisy_data[key], axis=0)
                for key in noisy_data.keys()
            }
   
    noisy_data['next_observation'] = noisy_data['next_observation'].reshape(-1, next_observation.shape[0])
    noisy_data['observation'] = noisy_data['observation'].reshape(-1, observation.shape[0])
    noisy_data['action'] = noisy_data['action'].reshape(-1, action.shape[0])


    with open(logdir, "wb") as f:
        pickle.dump(noisy_data, f) 
    print(f"Dataset created with {args.num_samples} samples and saved to {logdir}")

    
    

def train_sac(env, args):
    
    model = SAC("MlpPolicy", env, verbose=1, device = f"cuda:{args.devid}", seed=args.seed)
    model.learn(total_timesteps=args.steps, log_interval=4)
    savedir = os.path.join(args.log_dir, 'expert_models', f"sac_{args.env_name.split('-')[0]}")
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
    parser.add_argument("--noise_rate", type=float, help="Portion of samples to be noisy woth probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale", type=float, default=0.001, help="Standard deviation of the noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
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

    
    # create_dataset(env, args.num_samples)
    create_noisy_datasets(env, args)
    env.close()
    print()