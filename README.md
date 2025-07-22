# noisy_mujoco
Noisy MuJoCo environment generation for offline RL models.

**install the environment**
cd noisy_mujoco
conda env create -f environment.yaml
conda activate mopo
# Install viskit
git clone https://github.com/vitchyr/viskit.git
pip install -e viskit
pip install -e .

# Dataset Generation
**with our trained expert**
### create noiseless expert dataset ~1M samples
```
python create_dataset.py --num_samples 2200
```

### create noisy dataset with ~1M samples with action noise.
```
python create_dataset.py --num_samples 2200 --action --noise_rate_action 1 --scale_action 0.1
```

### create noisy dataset with ~1M samples with action and transition noise.
```
python create_dataset.py --num_samples 2200 --action --transition --noise_rate_action 1 --scale_action 0.1 --noise_rate_transition 1 --scale_transition 0.1
```

### create noisy dataset with ~1M samples with transition noise.
```
python create_dataset.py --num_samples 2200 --transition --noise_rate_transition 1 --scale_transition 0.1
```


**with D4RL trained expert**
### create noiseless expert dataset ~1M samples
```
python create_dataset.py --num_samples 1086 --farama
```

### create noisy dataset with ~1M samples with action noise.
```
python create_dataset.py --num_samples 1086 --action --noise_rate_action 1 --scale_action 0.1 --farama
```

### create noisy dataset with ~1M samples with action and transition noise.
```
python create_dataset.py --num_samples 1086 --action --transition --noise_rate_action 1 --scale_action 0.1 --noise_rate_transition 1 --scale_transition 0.1 --farama
```

### create noisy dataset with ~1M samples with transition noise.
```
python create_dataset.py --num_samples 1086 --transition --noise_rate_transition 1 --scale_transition 0.1 --farama
```




# Evaluate

### evaluate D4RL expert policy
```
python evaluate_policy.py --episodes 100 --action --transition --noise_rate_action 0.5 --noise_rate_transition 0.5 --scale_action 0.001 --scale_transition 0.001 --farama
```

### evaluate our expert policy
```
python evaluate_policy.py --episodes 100 --action --transition --noise_rate_action 0.5 --noise_rate_transition 0.5 --scale_action 0.001 --scale_transition 0.001
```


### train policy

```
python train.py --env_name Hopper-v2
```