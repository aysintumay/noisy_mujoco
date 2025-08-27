# Reinforcement Learning Environment for Abiomed World Model

This directory contains a reinforcement learning (RL) environment that wraps the trained Abiomed world model, allowing you to train RL agents to control p-levels (actions) for optimal physiological outcomes.

## Overview

The RL environment provides a standard gym-like interface for reinforcement learning algorithms. This allows for direct comparison between model-free RL approaches and model-based MPC approaches.

## Files

- `mpc.py`: Model predictive control baseline implementation
- `reward_func.py`: reward function implementation
- `config.py`: configs for different world model. 
- `rl_env.py`: Main RL environment implementation
- `rl_example.py`: Example script demonstrating usage 
- `requirements_rl.txt`: Full requirements for development

## Key Features

### Environment Interface
- **Observation Space**: Continuous vector of physiological parameters. The features are, in the pickle files of `/abiomed/downsampled/`.
```python
    ['PumpPressure', 'PumpSpeed', 'PumpFLow', 'LVP', 'LVEDP', 'SYSTOLIC','DIASTOLIC','PULSAT','PumpCurrent','Heart Rate', 'ESE_lv','Pump Level']
```
- **Action Space**: Discrete (9 actions, p-levels 2-10) or Continuous (*normalized* p-levels)

To use reward shaping; set gamma1, gamma2 or gamma3 a nonzero value. Example environment call with reward shaping turned on:
```
env = AbiomedRLEnvFactory.create_env(
									model_name="10min_1hr_all_data",
									model_path=None,
									data_path=None,
									max_steps=6,
									gamma1=0.1,
									gamma2=0.1,
									gamma3=0.1,
									action_space_type="continuous",
									reward_type="smooth",
									normalize_rewards=True,
									seed=42,
									)
```

- **Reward Function**: Smooth reward function currently only evaluating MAP, heart rate, pulsatility.
- **Episode Structure**: Configurable episode length with automatic episode generation from test data. Default is 24 hours.

### Env Factory Pattern
The `AbiomedRLEnvFactory` provides a convenient way to create environments with different configurations:
- Model selection (10min_1hr_window, 5min_2hr_window, etc.)
- Action space type (discrete/continuous)
- Reward function type (smooth/discrete)
- Episode length and reward normalization
- Noisy observation (use `noise_rate` $\in [0,1]$ and `noise_scale` for Gaussian noise.)

## Installation


### Manual Installation
```bash
# Or install full requirements
pip install -r requirements_rl.txt
```

## Usage


### RL Training and Evaluation

Use the provided example script for training and evaluation:

```bash
# Train a random policy
python rl_example.py --policy_type random --train_episodes 1 --eval_episodes 24

# Train an SAC policy (continuous)
python rl_example.py --policy_type sac --train_episodes 10000 --batch_size 64 --max_steps 24

# Train a PPO policy (discrete)
python rl_example.py --policy_type ppo --train_episodes 10000 --batch_size 64 --max_steps 24

```

### MPC training 

```bash
# Compare with MPC baseline (1 step greedy planning works best)
python mpc.py --planner_type mean --n_scenarios 50 --horizon 1 --plans 10


# MPC with CVaR rewards
python mpc.py --planner_type cvar --n_scenarios 50 --horizon 3 --plans 50 --cvar_alpha 0.2 --cvar_samples 20 
```

### Monitoring

The following command allows for monitoring training progress with tensor board.

```bash
tensorboard --logdir sb3_log/
```

### Integration with RL Libraries

The environment is compatible with popular RL libraries:

```python
# Stable Baselines3
from stable_baselines3 import PPO
from rl_env import AbiomedRLEnvFactory

env = AbiomedRLEnvFactory.create_env()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Create offline datasets

To create offline datasets in D4RL format, simply run

`python sample_offline_dataset.py`

to sample noisy observation, run

`python sample_offline_dataset.py --noise_rate 0.8 --noise_scale 0.2 --num_episodes 100000`

