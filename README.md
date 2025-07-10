# noisy_mujoco
Noisy MuJoCo environment generation for offline RL models.

## create noisy dataset with 1M samples with action and transition noise.
```
python create_dataset.py --env_name Hopper-v2 --action --transition --num_samples 1000000
```

## create noisy dataset with 1M samples with action noise.
```
python create_dataset.py --env_name Hopper-v2 --action --num_samples 1000000
```

## train policy

```
python train.py --env_name Hopper-v2
```