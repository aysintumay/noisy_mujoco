# --- Model Predictive Control (MPC) for planning with best reward ---
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import pickle
import json
import sys

from reward_func import compute_reward_smooth
from model import WorldModel
import config

def mpc_planning(
    world_model, x, planning_horizon=1, n_plans=50, n_steps=5
):
    """a
    Model Predictive Control (MPC) loop.
    Args:
        world_model: your trained world model
        init_batch_data: initial input data for simulation
        device: torch device
        forecast_horizon: planning horizon for each MPC step
        total_horizon: total steps to simulate
        n_plans: number of random plans to sample at each step
        n_steps: number of MPC steps to run
    Returns:
        actions_taken, outputs
    """

    def action_sampler(length):
        # randomly integer between 2 to 10
        pl = [torch.randint(2, 10, (1,)).item() for _ in range(length)]
        return pl

    def reward_function_smooth(world_model, output):
        rewards = [
            compute_reward_smooth(world_model.unnorm_output(output[i]))
            for i in range(len(output))
        ]
        sum_rewards = sum(rewards)
        return sum_rewards
    
    actions_taken = []
    outputs = []
    input_state = x
    for step in range(n_steps):
        best_reward = -float("inf")
        best_action_seq = None
        best_output = None
        for i in range(n_plans):

            # Sample a random action sequence
            action_seq = action_sampler(planning_horizon)

            # Simulate outcome
            # output = world_model.step(x, action_seq).detach()
            output = world_model.sample_autoregressive(
                input_state, planning_horizon, custom_pl=action_seq
            )
            reward = reward_function_smooth(world_model, output)
            #print("action_seq:", action_seq, "reward:", reward)
            if reward > best_reward:
                best_reward = reward
                best_action_seq = action_seq
                best_output = output
                input_state = output[0]

        #print("best_action_seq:", best_action_seq, "best_reward:", best_reward)

        # Apply the first action in the best sequence
        actions_taken.append(best_action_seq[0])
        outputs.append(best_output[0])
    return actions_taken, outputs

def mpc_planning_cvar(
    world_model, x, planning_horizon=1, n_plans=50, n_steps=5, cvar_alpha=0.2, n_samples=10
):
    """
    Model Predictive Control (MPC) loop using CVaR as the planning objective.
    Args:
        world_model: your trained world model
        x: initial input data for simulation
        planning_horizon: planning horizon for each MPC step
        n_plans: number of random plans to sample at each step
        n_steps: number of MPC steps to run
        cvar_alpha: fraction of lowest rewards to average for CVaR (e.g., 0.2)
        n_samples: number of stochastic rollouts per plan
    Returns:
        actions_taken, outputs
    """
    def action_sampler(length):
        # randomly integer between 2 to 10
        pl = [torch.randint(2, 10, (1,)).item() for _ in range(length)]
        return pl

    def reward_function_cvar(world_model, input_state, action_seq, planning_horizon, cvar_alpha, n_samples):
        # Sample multiple rollouts
        outputs = world_model.sample_autoregressive_multiple(
            input_state, planning_horizon, custom_pl=action_seq, sample_size=n_samples
        )
        # Compute sum of rewards for each rollout
        rewards = []
        for output in outputs:
            rollout_rewards = [
                compute_reward_smooth(world_model.unnorm_output(output[i]))
                for i in range(len(output))
            ]
            rewards.append(sum(rollout_rewards))
        rewards = np.array(rewards)
        # Compute CVaR (average of the lowest alpha fraction)
        cutoff = max(1, int(np.ceil(cvar_alpha * len(rewards))))
        cvar = np.mean(np.sort(rewards)[:cutoff])
        return cvar

    actions_taken = []
    outputs = []
    input_state = x
    for step in range(n_steps):
        best_cvar = -float("inf")
        best_action_seq = None
        best_output = None
        for i in range(n_plans):
            # Sample a random action sequence
            action_seq = action_sampler(planning_horizon)
            # Compute CVaR for this action sequence
            cvar = reward_function_cvar(
                world_model, input_state, action_seq, planning_horizon, cvar_alpha, n_samples
            )
            if cvar > best_cvar:
                best_cvar = cvar
                best_action_seq = action_seq
                # For output, just sample one rollout for the next state
                best_output = world_model.sample_autoregressive(
                    input_state, planning_horizon, custom_pl=action_seq
                )
                input_state = best_output[0]
        actions_taken.append(best_action_seq[0])
        outputs.append(best_output[0])
    return actions_taken, outputs


def run_mpc(
    world_model,
    planning_shootout_horizon=3,
    n_plans=50,
    n_steps=5,
    n_scenarios=10,
    results_path=None,
    planner_type="mean",
    cvar_alpha=0.2,
    cvar_samples=10,
):
    """
    planner_type: 'mean' or 'cvar'
    cvar_alpha, cvar_samples: only used if planner_type == 'cvar'
    """
    if planner_type == "mean":
        planner = mpc_planning
        planner_kwargs = {}
    elif planner_type == "cvar":
        planner = mpc_planning_cvar
        planner_kwargs = {"cvar_alpha": cvar_alpha, "n_samples": cvar_samples}
    else:
        print(f"Unknown planner_type: {planner_type}")
        sys.exit(1)

    planning_rewards = []
    original_rewards = []
    original_model_rewards = []

    our_actions = []
    original_actions = []
    all_outputs = []
    all_gt = []

    random_indices = np.random.randint(0, 560-n_steps, size=n_scenarios)
    print(f"sampling {n_scenarios} random indices from {560-n_steps} possible indices")

    for i in tqdm(random_indices):

        start_idx = i
        end_idx = start_idx + n_steps + 1

        batch_data = []
        test_loader = DataLoader(world_model.data_test, batch_size=1, shuffle=False)
        loader = iter(test_loader)
        for i in range(end_idx):
            if i <= start_idx:
                example = next(loader)
                continue
            example = next(loader)
            batch_data.append(example)

        x = batch_data[0][0]
        actions_taken, outputs = planner( 
            world_model, x, planning_horizon=planning_shootout_horizon, 
            n_plans=n_plans, n_steps=n_steps, **planner_kwargs
        )
        our_actions.append(actions_taken)

        # calculate rewards
        unnorm_outputs = [world_model.unnorm_output(outputs[i]) for i in range(len(outputs))]
        rewards = [compute_reward_smooth(unnorm_outputs[i]) for i in range(len(unnorm_outputs))]
        sum_rewards = sum(rewards)
        
        unnorm_y = world_model.unnorm_y(batch_data[:n_steps])
        original_reward = [compute_reward_smooth(unnorm_y[i]) for i in range(len(unnorm_y))]
        sum_original_reward = sum(original_reward) 
   
        # calculate original action model rewards
        gt_action = [batch_data[i][1] for i in range(len(batch_data))]
        original_model_outputs = world_model.sample_autoregressive(
            x, n_steps, custom_pl=gt_action
        )
        original_model_reward = [
            compute_reward_smooth(world_model.unnorm_output(original_model_outputs[i]))
            for i in range(len(original_model_outputs))
        ]
        sum_original_model_reward = sum(original_model_reward)

        if "1hr" in results_path :
            sum_rewards = sum_rewards / 2
            sum_original_reward = sum_original_reward / 2
            sum_original_model_reward = sum_original_model_reward / 2
        
        all_outputs.append(unnorm_outputs)
        all_gt.append(unnorm_y)
        planning_rewards.append(sum_rewards)
        original_rewards.append(sum_original_reward)
        original_model_rewards.append(sum_original_model_reward)

        # unnorm gt action for comparison
        gt_action_unnorm = torch.cat(gt_action, dim=0) * world_model.std[12] + world_model.mean[12]
        # get majority vote of each row
        gt_action_voted = torch.mode(gt_action_unnorm, dim=1)[0].numpy()
        #print("gt_action[0]:", gt_action_voted, "original rewards:", original_rewards)

        original_actions.append(gt_action_voted)

    # calculate metrics
    our_actions = np.array(our_actions)
    original_actions = np.array(original_actions)

    print(
        f"planning reward mean: {np.mean(planning_rewards):.2f}",
        f"+/- {np.std(planning_rewards):.2f}",
    )
    print(
        f"original reward mean: {np.mean(original_rewards):.2f}",
        f"+/- {np.std(original_rewards):.2f}",
    )
    print(
        f"original model reward mean: {np.mean(original_model_rewards):.2f}",
        f"+/- {np.std(original_model_rewards):.2f}",
    )

    mae = np.sum(np.abs(our_actions - original_actions)) / (
        len(our_actions) * len(our_actions[0])
    )
    print(f"MAE: {mae:.2f}")
    
    # write results to file
    metrics = {}

    results = {"outputs": all_outputs,
            "origional_outputs": all_gt,
            "actions": our_actions, 
            "original_actions": original_actions, 
            "rewards": planning_rewards, 
            "original_rewards": original_rewards, 
            "original_model_rewards": original_model_rewards, 
            "mae": mae}

    if results_path != "":
        with open(results_path+".json", "a") as f:
            metrics["planning_reward_mean"] = np.mean(planning_rewards).astype(float)
            metrics["planning_reward_std"] = np.std(planning_rewards).astype(float)
            metrics["original_reward_mean"] = np.mean(original_rewards).astype(float)
            metrics["original_reward_std"] = np.std(original_rewards).astype(float)
            metrics["original_model_reward_mean"] = np.mean(original_model_rewards).astype(float)
            metrics["original_model_reward_std"] = np.std(original_model_rewards).astype(float)
            metrics["mae"] = mae.astype(float)
            
            json.dump(metrics, f)

        with open(results_path+".pkl", "wb") as f:
            pickle.dump(results, f)

    return batch_data, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="10min_1hr_window")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--plans", type=int, default=15)
    parser.add_argument("--n_scenarios", type=int, default=100)
    parser.add_argument("--planner_type", type=str, default="mean", choices=["mean","cvar"])
    parser.add_argument("--cvar_alpha", type=float, default=0.2)
    parser.add_argument("--cvar_samples", type=int, default=20)
    args = parser.parse_args()

    model_kwargs = config.model_configs[args.model_name]
    experiment_name = f"{args.model_name}_{args.steps}steps_{args.plans}plans"
    results_path = f"/abiomed/downsampled/results/{experiment_name}_{args.planner_type}"
    world_model = WorldModel(**model_kwargs)

    # model_path = "../../data2025/models/10min_2hr_window_model.pth"
    model_path = (
        f"/abiomed/downsampled/models/{args.model_name}_model.pth"
    )
    world_model.load_model(model_path)
    # data_path = "../../data2025/10min_2hr_window.pkl"
    data_path = f"/abiomed/downsampled/{args.model_name}.pkl"
    world_model.load_data(data_path)

    run_mpc(
        world_model,
        planning_shootout_horizon=args.horizon,
        n_plans=args.plans,
        n_steps=args.steps,
        n_scenarios=args.n_scenarios,
        results_path=results_path,
        planner_type=args.planner_type,
        cvar_alpha=args.cvar_alpha,
        cvar_samples=args.cvar_samples,
    )


if __name__ == "__main__":
    main()