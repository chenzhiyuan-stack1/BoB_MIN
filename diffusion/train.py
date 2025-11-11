# exp对应的name和产物保存的dir就是可以以训练的时间来命名yymmddhhmmss
# 训练可以参考/home/min414/data2/BoB_MIN/diffusion/main_e2e_BoB.py
# 打点要详细一些，方便后续分析，可以结合wandb等工具
# IS_TRAINING = True判断离线还是在线

# 如果离线训练，只用从指定的路径加载一次样本，然后多轮训练，隔一段时间落ckpt，并在设定的small_evaluation_datasets上eval，基本可以完全参考main_e2e_BoB.py的流程

# 如果在线训练，则需要在每轮采集数据后，重新加载数据集，然后进行多次更新
# exp的第i轮迭代
# 呼测、采集数据
# receive_with_tc_train.sh 脚本会调用agent.save_newest_model()落的ckpt进行呼测
# id结合exp和i的信息来命名，最好方便从中解析exp和i
# 采集完之后，进行数据集转换和训练（可以修改一下receive_with_tc_train.sh的输出/返回，来判断采集完）
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
from typing import Dict

import numpy as np
import torch
import wandb

# 确保可以导入项目中的其他模块
# 将项目根目录添加到Python路径
# sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.ql_diffusion_e2e import Diffusion_QL as Agent
from diffusion.utils.data_sampler import ReplayBuffer
from diffusion.utils.logger import logger, setup_logger
from diffusion.utils import utils as diffusion_utils
from diffusion.norm_vector import adjust_dataset
from diffusion.eval import evaluate_policy
from dataset.result2dataset import process_results_to_dataset, save_dataset_as_pickle
from utils.calculate_trace import calculate_metrics_for_collection

STATE_DIM = 66
ACTION_DIM = 1 
MAX_ACTION = 20

def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def log_all_metrics(step: int, train_metrics: Dict[str, list], mse: float, accuracy: float, over: float, prefix: str, trace_metrics: Dict = None):
    """Helper to log all aggregated training and evaluation metrics to console and WandB."""
    avg_train_metrics = {key: np.mean(val) for key, val in train_metrics.items()}

    diffusion_utils.print_banner(f"{prefix} Step: {step}", separator="*", num_star=90)
    logger.record_tabular(f'{prefix}/Step', step)
    # 记录训练指标
    for key, val in avg_train_metrics.items():
        logger.record_tabular(f'Train/{key}', val)
    # 记录评估指标
    logger.record_tabular('Eval/MSE', mse)
    logger.record_tabular('Eval/Accuracy', accuracy)
    logger.record_tabular('Eval/Overestimation', over)
    # 记录在线交互轨迹指标
    if trace_metrics:
        for key, val in trace_metrics.items():
            # 排除非数值型指标
            if isinstance(val, (int, float)):
                logger.record_tabular(f'Trace/{key}', val)
    logger.dump_tabular()

    if wandb.run:
        wandb_log_data = {}
        for key, val in avg_train_metrics.items():
            wandb_log_data[f'{prefix}/Train/{key}'] = val
        wandb_log_data[f'{prefix}/Eval/MSE'] = mse
        wandb_log_data[f'{prefix}/Eval/Accuracy'] = accuracy
        wandb_log_data[f'{prefix}/Eval/Overestimation'] = over
        
        if trace_metrics:
            for key, val in trace_metrics.items():
                if isinstance(val, (int, float)):
                    wandb_log_data[f'{prefix}/Trace/{key}'] = val
        wandb.log(wandb_log_data, step=step)

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
            replay_buffer.add_transitions(dataset)
        del dataset
    print(f"Replay buffer size: {replay_buffer._size}")
    
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

        # Evaluation
        mse, accuracy, over = evaluate_policy(agent.actor, args.eval_datasets, args.device)

        # Log metrics
        log_all_metrics(curr_epoch, loss_metric, mse, accuracy, over, prefix="Offline")
        
        # Save model
        agent.save_model(args.exp_run_path, curr_epoch)
        
    # Final model save
    agent.save_model(args.exp_run_path, "final_offline")

def run_online_training(args: argparse.Namespace, agent: Agent, replay_buffer: ReplayBuffer):
    """Online training loop: collect -> process -> train."""
    diffusion_utils.print_banner("Starting ONLINE Training", separator="=", num_star=90)

    # Init actor and critic from final ckpt in offline training
    # args.offline_name设定为离线训练的args.name
    # args.name = f"{timestamp}_{args.name}_{args.mode}_{unique_id}"
    # offline_path相当于离线训练时的args.exp_run_path
    # args.exp_run_path = os.path.join(args.exp_output_dir, args.name)
    offline_path = os.path.join(args.exp_output_dir, args.offline_name)
    agent.load_model(offline_path, "final_offline")
    
    for i in range(args.online_rounds):
        diffusion_utils.print_banner(f"Online Round {i + 1} / {args.online_rounds}", separator="=", num_star=90)

        # --- Step 1: Data Collection ---
        agent.save_newest_model(args.online_model_path) # for data collection in receive_with_tc_train.sh
        
        # sync model to send
        # local destination: f'{args.online_model_path}/MDQL.pth'
        # remote destination: ssh -p 2223 knw@202.120.36.216 "cd BoB_MIN" f'{args.online_model_path}/MDQL.pth'
        print("Syncing model to the remote sender...")
        local_model_path = os.path.join(args.online_model_path, 'MDQL.pth')
        remote_user_host = "knw@202.120.36.216"
        remote_base_dir = "BoB_MIN"
        # The destination file on the remote server will be named 'MDQL.pth'
        remote_dest_path = f"{remote_base_dir}/{args.online_model_path}/MDQL.pth"
        try:
            # Construct the scp command
            scp_command = [
                "scp",
                "-P", "2223",  # Note: scp uses uppercase -P for port
                local_model_path,
                f"{remote_user_host}:{remote_dest_path}"
            ]
            sync_result = subprocess.run(
                scp_command,
                check=True,  # This will raise an exception if scp fails
                capture_output=True,
                text=True
            )
            print("Model successfully synced to remote sender.")
        except FileNotFoundError:
            print(f"ERROR: 'scp' command not found. Please ensure it's installed and in your PATH.")
            continue
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to sync model to remote sender.")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Stderr: {e.stderr}")
            continue
        
        collection_id = f"{Path(args.name).stem}_round_{i}"
        
        print(f"Running data collection script for ID {collection_id} ...")
        with subprocess.Popen(
            ["bash", "receive_with_tc_train.sh", collection_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as proc:
            output_lines = []
            for line in proc.stdout:
                print(line, end='')  # 实时打印每一行
                output_lines.append(line)
            proc.wait()
            result_stdout = ''.join(output_lines)
            
        if "__DATA_COLLECTION_DONE__" in result_stdout:
            print("Data collection finished successfully.")
        else:
            print(f"WARNING: Data collection script did not finish as expected for ID {collection_id}.")
            # print(f"Stdout: {result_stdout}")
            # print(f"Stderr: {result.stderr}")
            continue

        # --- Step 2: Data Processing ---
        print("Processing collected data into a dataset...")
        trace_metrics = None # 初始化
        try:
            collection_folder_path = os.path.join(args.results_basedir, collection_id)
            new_dataset = process_results_to_dataset(args.results_basedir, [collection_id])
            if new_dataset['observations'].shape[0] == 0:
                print("Warning: No data points processed from this collection round. Skipping training.")
                continue
            pickle_name = f"/home/min414/data2/extra_storage/processed_dataset_{collection_id}.pickle"
            save_dataset_as_pickle(new_dataset, pickle_name)
            print(f"New dataset saved to {pickle_name}")
            
            print("Calculating performance metrics for the collected trace...")
            # process_one_trace输入的folder_path是collection_folder_path下的一层目录
            # 这里要遍历collection_folder_path下的所有trace文件夹，计算每个trace的指标，然后取平均
            # 参考utils/calculate_trace.py中的main函数
            # --- 核心改动：调用新函数计算平均性能指标 ---
            trace_metrics = calculate_metrics_for_collection(collection_folder_path)
            if trace_metrics:
                print("Average trace metrics calculated successfully.")
            else:
                print("Warning: Failed to calculate trace metrics for this round.")
        except Exception as e:
            print(f"ERROR: Failed to process dataset for {collection_id}. Error: {e}")
            continue

        # --- Step 3: Update Replay Buffer ---
        print("Adding new data to the replay buffer...")
        new_dataset = adjust_dataset(new_dataset)
        replay_buffer.add_transitions(new_dataset)
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
        mse, accuracy, over = evaluate_policy(agent.actor, args.eval_datasets, args.device)

        # Log metrics
        log_all_metrics(i, loss_metric, mse, accuracy, over, prefix="Online", trace_metrics=trace_metrics)
        
        # --- Step 6: Save Model ---
        agent.save_model(args.exp_run_path, f"online_round_{i}") # for safety backup

    # Final model save after all rounds
    agent.save_model(args.exp_run_path, "final_online")
    print("Online training loop finished.")



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
    parser.add_argument('--dataset_paths', type=str, nargs='+', default=["BoB_012.pickle", "BoB_45.pickle",], help="Paths to offline dataset files")
    parser.add_argument('--eval_freq', type=int, default=10, help="Evaluation frequency in training iterations")
    parser.add_argument('--num_epochs', type=int, default=4000)
    parser.add_argument('--num_steps_per_epoch', type=int, default=1000)

    # --- Online Mode Specific ---
    parser.add_argument('--online_rounds', type=int, default=10, help="Total number of collect-train cycles")
    parser.add_argument('--iterations_per_round', type=int, default=1000, help="Training iterations after each data collection")
    parser.add_argument('--results_basedir', type=str, default="results", help="Base directory for raw log data")
    parser.add_argument('--online_model_path', type=str, default="model", help="Base directory for online model checkpoints")
    parser.add_argument('--offline_name', type=str, default="251105234045_offline_exp_offline_400a", help="Offline experiment name for initializing online training")

    # --- Agent & RL Parameters ---
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--buffer_size', type=int, default=3_000_000)
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
