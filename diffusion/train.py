# exp对应的name和产物保存的dir就是可以以训练的时间来命名yymmddhhmmss
# 训练可以参考/home/min414/data2/BoB_MIN/diffusion/main_e2e_BoB.py
# 打点要详细一些，方便后续分析，可以结合wandb等工具
# IS_TRAINING = True判断离线还是在线

# 如果离线训练，只用从指定的路径加载一次样本，然后多轮训练，隔一段时间落ckpt，并在设定的small_evaluation_datasets上eval，基本可以完全参考main_e2e_BoB.py的流程

# 如果在线训练，则需要在每轮采集数据后，重新加载数据集，然后进行多次更新
# exp的第i轮迭代
# 呼测、采集数据
# receive_with_tc.sh 脚本会调用agent.save_newest_model()落的ckpt进行呼测
# id结合exp和i的信息来命名，最好方便从中解析exp和i
# 采集完之后，进行数据集转换和训练（可以修改一下receive_with_tc.sh的输出/返回，来判断采集完）
# 原始日志数据落在results/id 路径下
# 参考dataset/result2dataset.py，调用里面的process_results_to_dataset、save_dataset_as_pickle将结果转换为数据集
# 加载当前一轮的数据集
# 基于这一波的数据集更新iterations轮
# 保存模型

import os
import sys
import uuid
import pickle
import random
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import List
from tqdm import tqdm

import numpy as np
import torch
import wandb

# 确保可以导入项目中的其他模块
# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from diffusion.ql_diffusion_e2e import Diffusion_QL as Agent
from diffusion.utils.data_sampler import ReplayBuffer
from diffusion.utils.logger import logger, setup_logger
from diffusion.utils import utils as diffusion_utils
from dataset.result2dataset import process_results_to_dataset, save_dataset_as_pickle
from norm_vector import adjust_dataset, NORMAL_VECTOR

STATE_DIM = 66
ACTION_DIM = 1 
MAX_ACTION = 20

def evaluate_RtcBwp(policy_fn, eval_dataset: list, device: str):
    every_call_mse = []
    every_call_accuracy = []
    every_call_over = []
    for f_path in tqdm(eval_dataset, desc="Overall Evaluation Progress", position=0):
        with open(f_path, 'rb') as f:
            call_data = pickle.load(f)

        normal_vector = NORMAL_VECTOR
        observations = np.asarray(call_data["observations"], dtype=np.float32)
        true_capacity = np.asarray(call_data["true_capacities"], dtype=np.float32)
        next_observations = np.asarray(call_data["next_observations"], dtype=np.float32)
        actions = np.asarray(call_data["actions"], dtype=np.float32)
        rewards = np.asarray(call_data["rewards"], dtype=np.float32)
        
        model_predictions = []
        # 为内部循环添加tqdm，leave=False使其完成后消失，避免刷屏
        for t in tqdm(range(observations.shape[0]), desc=f"Processing {os.path.basename(f_path)}", position=1, leave=False):
            obss = observations[t : t + 1, :].reshape(-1)
            obss_ = obss * normal_vector
            obss_ = torch.tensor(obss_.reshape(1, -1), device=device, dtype=torch.float32)
            action = actions[t:t+1, :].reshape(-1)
            action = action / 1e6
            action_tensor = torch.tensor(action.reshape(1, -1), dtype=torch.float32).to(device)
            reward = rewards[t:t+1].reshape(-1)
            reward_tensor = torch.tensor(reward.reshape(1, -1), dtype=torch.float32).to(device)
            next_obss = next_observations[t:t+1, :].reshape(-1)
            next_obss = next_obss * normal_vector
            next_obss_tensor = torch.tensor(next_obss.reshape(1, -1), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action = policy_fn.sample_action(obss_) # bps
                action = action / 1e6
            bw_prediction = np.squeeze(action)
            model_predictions.append(bw_prediction)
        # mse and accuracy of this call
        model_predictions = np.asarray(model_predictions, dtype=np.float32)
        true_capacity = true_capacity / 1e6
        # model_predictions = model_predictions / 1e6
        call_mse = []
        call_accuracy = []
        call_over = []
        for true_bw, pre_bw in zip(true_capacity, model_predictions):
            if np.isnan(true_bw) or np.isnan(pre_bw):
                continue
            else:
                mse_ = (true_bw - pre_bw) ** 2
                call_mse.append(mse_)
                accuracy_ = max(0, 1 - abs(pre_bw - true_bw) / true_bw)
                call_accuracy.append(accuracy_)
                over = max(0,(pre_bw - true_bw) / true_bw)
                call_over.append(over)
        call_mse = np.asarray(call_mse, dtype=np.float32)
        every_call_mse.append(np.mean(call_mse))
        call_accuracy = np.asarray(call_accuracy, dtype=np.float32)
        every_call_accuracy.append(np.mean(call_accuracy))
        call_over = np.asarray(call_over, dtype=np.float32)
        every_call_over.append(np.mean(call_over))
    every_call_mse = np.asarray(every_call_mse, dtype=np.float32)
    every_call_accuracy = np.asarray(every_call_accuracy, dtype=np.float32)
    every_call_over = np.asarray(every_call_over, dtype=np.float32)
    return np.mean(every_call_mse), np.mean(every_call_accuracy), np.mean(every_call_over)

def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def run_offline_training(args: argparse.Namespace, agent: Agent, replay_buffer: ReplayBuffer):
    """Offline training loop."""
    diffusion_utils.print_banner("Starting OFFLINE Training", separator="=", num_star=90)

    # Load initial dataset(s)
    for idx, path in enumerate(args.dataset_paths):
        print(f"Loading dataset from: {path}")
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        dataset = adjust_dataset(dataset)
        if idx == 0:
            replay_buffer.load_dataset(dataset)
        else:
            replay_buffer.add_transition(dataset)
        del dataset
    print(f"Replay buffer size: {replay_buffer._size}")

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    diffusion_utils.print_banner(f"Training Start", separator="*", num_star=90)
    while training_iters < max_timesteps:
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(
            replay_buffer,
            iterations=iterations,
            batch_size=args.batch_size,
        )
        training_iters += iterations
        curr_epoch = int(training_iters // args.num_steps_per_epoch)

        # Logging
        diffusion_utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        log_metrics(curr_epoch, loss_metric)

        # Evaluation
        mse, accuracy, over = evaluate_RtcBwp(agent, args.eval_datasets, args.device)
        evaluations.append([mse, accuracy, over, np.mean(loss_metric['bc_loss']), curr_epoch])
        np.save(os.path.join(args.exp_run_path, "eval_results.npy"), evaluations)
        log_evaluation(mse, accuracy, over)

        # Save model
        agent.save_model(args.exp_run_path, curr_epoch)

def run_online_training(args: argparse.Namespace, agent: Agent, replay_buffer: ReplayBuffer):
    """Online training loop: collect -> process -> train."""
    diffusion_utils.print_banner("Starting ONLINE Training", separator="=", num_star=90)

    for i in range(args.online_rounds):
        diffusion_utils.print_banner(f"Online Round {i + 1} / {args.online_rounds}", separator="=", num_star=90)

        # --- Step 1: Data Collection ---
        agent.save_newest_model(args.exp_run_path)
        
        collection_id = f"{Path(args.name).stem}_round_{i}"
        
        print(f"Starting data collection with ID: {collection_id}")
        try:
            subprocess.run(["bash", "receive_with_tc.sh", collection_id], check=True, capture_output=True, text=True)
            print("Data collection finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Data collection script failed for ID {collection_id}.")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            continue

        # --- Step 2: Data Processing ---
        print("Processing collected data into a dataset...")
        try:
            new_dataset = process_results_to_dataset(args.results_basedir, [collection_id])
            if new_dataset['observations'].shape[0] == 0:
                print("Warning: No data points processed from this collection round. Skipping training.")
                continue
        except Exception as e:
            print(f"ERROR: Failed to process dataset for {collection_id}. Error: {e}")
            continue

        # --- Step 3: Update Replay Buffer ---
        print("Adding new data to the replay buffer...")
        new_dataset = adjust_dataset(new_dataset)
        replay_buffer.load_dataset(new_dataset)
        del new_dataset
        print(f"Replay buffer new size: {replay_buffer.size}")

        # --- Step 4: Training ---
        print(f"Training for {args.iterations_per_round} iterations...")
        loss_metric = agent.train(
            replay_buffer,
            iterations=args.iterations_per_round,
            batch_size=args.batch_size,
        )

        # --- Step 5: Logging & Evaluation ---
        log_metrics(i, loss_metric, prefix="Online")
        mse, accuracy, over = evaluate_RtcBwp(agent, args.eval_datasets, args.device)
        log_evaluation(mse, accuracy, over, prefix="Online")

    # Final model save after all rounds
    agent.save_model(args.exp_run_path, "final_online")
    print("Online training loop finished.")

def log_metrics(epoch: int, loss_metric: dict, prefix: str = "Offline"):
    """Helper to log training metrics."""
    bc_loss = np.mean(loss_metric['bc_loss'])
    actor_loss = np.mean(loss_metric['actor_loss'])
    v_loss = np.mean(loss_metric['v_loss'])
    ql_loss = np.mean(loss_metric['ql_loss'])

    logger.record_tabular(f'{prefix}/Trained Epochs', epoch)
    logger.record_tabular(f'{prefix}/BC Loss', bc_loss)
    logger.record_tabular(f'{prefix}/Actor Loss', actor_loss)
    logger.record_tabular(f'{prefix}/V Loss', v_loss)
    logger.record_tabular(f'{prefix}/QL Loss', ql_loss)
    logger.dump_tabular()

    if wandb.run:
        wandb.log({
            f'{prefix}/BC Loss': bc_loss,
            f'{prefix}/Actor Loss': actor_loss,
            f'{prefix}/V Loss': v_loss,
            f'{prefix}/QL Loss': ql_loss,
        }, step=epoch)

def log_evaluation(mse: float, accuracy: float, over: float, prefix: str = "Eval"):
    """Helper to log evaluation metrics."""
    logger.record_tabular(f'{prefix}/MSE', mse)
    logger.record_tabular(f'{prefix}/Accuracy', accuracy)
    logger.record_tabular(f'{prefix}/Overestimation', over)
    logger.dump_tabular()

    if wandb.run:
        wandb.log({
            f'{prefix}/MSE': mse,
            f'{prefix}/Accuracy': accuracy,
            f'{prefix}/Overestimation': over,
        })

def main(args: argparse.Namespace):
    # --- 1. Setup ---
    set_seed(args.seed)
    
    # Generate a unique name and path for the experiment run
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:4]
    # We modify the 'name' and add 'exp_run_path' to the args object for easy access
    args.name = f"{timestamp}_{args.name}_{args.mode}_{unique_id}"
    args.exp_run_path = os.path.join(args.exp_output_dir, args.name)
    
    os.makedirs(args.exp_run_path, exist_ok=True)

    # Setup logging
    if args.use_wandb:
        wandb.init(
            project=args.project,
            group=args.group,
            name=args.name,
            config=args,
            sync_tensorboard=True
        )
    variant = vars(args)
    setup_logger(os.path.basename(args.name), variant=variant, log_dir=args.exp_run_path)
    diffusion_utils.print_banner(f"Saving all outputs to: {args.exp_run_path}")

    # --- 2. Initialize Agent and Replay Buffer ---
    agent = Agent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        device=args.device,
        discount=args.discount,
        tau=args.tau,
        max_q_backup=args.max_q_backup,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        eta=args.eta,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_maxt=args.num_epochs,
        grad_norm=args.grad_norm,
    )

    replay_buffer = ReplayBuffer(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        buffer_size=args.buffer_size,
    )

    # --- 3. Execute Training Mode ---
    if args.mode == 'offline':
        run_offline_training(args, agent, replay_buffer)
    elif args.mode == 'online':
        run_online_training(args, agent, replay_buffer)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.use_wandb:
        wandb.finish()
    print("--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- Experiment Settings ---
    parser.add_argument('--project', type=str, default="BoB-DiffusionQL-Training", help="WandB project name")
    parser.add_argument('--group', type=str, default="default-group", help="WandB group name")
    parser.add_argument('--name', type=str, default="train", help="Base name for the experiment")
    parser.add_argument('--mode', type=str, default="offline", choices=["offline", "online"], help="Training mode")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--exp_output_dir', type=str, default="results_exp", help="Unified directory for all experiment outputs")
    parser.add_argument('--use_wandb', action='store_true', help="Enable WandB logging")

    # --- Offline Mode Specific ---
    parser.add_argument('--dataset_paths', type=str, nargs='+', default=["BoB_012.pickle"], help="Paths to offline dataset files")
    parser.add_argument('--eval_freq', type=int, default=10, help="Evaluation frequency in training iterations")
    parser.add_argument('--num_epochs', type=int, default=4000)
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000)

    # --- Online Mode Specific ---
    parser.add_argument('--online_rounds', type=int, default=10, help="Total number of collect-train cycles")
    parser.add_argument('--iterations_per_round', type=int, default=10000, help="Training iterations after each data collection")
    parser.add_argument('--results_basedir', type=str, default="results", help="Base directory for raw log data")

    # --- Agent & RL Parameters ---
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--buffer_size', type=int, default=2_000_000)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', action='store_true', help="Enable learning rate decay")
    parser.add_argument('--grad_norm', type=float, default=10.0)

    # --- Diffusion Parameters ---
    parser.add_argument('--beta_schedule', type=str, default='vp')
    parser.add_argument('--n_timesteps', type=int, default=10)
    parser.add_argument('--eta', type=float, default=10.0, help="BC strength")
    parser.add_argument('--max_q_backup', action='store_true', help="Use max Q backup")

    # --- Evaluation ---
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=["BoB_3.pickle"], help="Paths to evaluation dataset files")

    args = parser.parse_args()
    main(args)
