import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from diffusion.norm_vector import NORMAL_VECTOR

def evaluate_trace(eval_dataset_paths: list):
    """
    评估数据集中记录的 actions (通常是来自真实轨迹或另一个模型的输出)。
    不使用策略模型进行推理。
    """
    every_call_mse = []
    every_call_accuracy = []
    every_call_over = []

    progress_bar = tqdm(eval_dataset_paths, desc="Initializing trace evaluation...", position=0, leave=True)

    for f_path in progress_bar:
        progress_bar.set_description(f"Processing {os.path.basename(f_path)}")

        with open(f_path, 'rb') as f:
            call_data = pickle.load(f)

        true_capacity = np.asarray(call_data["true_capacities"], dtype=np.float32)
        actions = np.asarray(call_data["actions"], dtype=np.float32).flatten()

        pred_bw = actions / 1e6  # 假设 actions 单位是 bps
        true_bw = true_capacity / 1e6

        valid_indices = ~np.isnan(true_bw) & ~np.isnan(pred_bw) & (true_bw > 0)
        if not np.any(valid_indices):
            tqdm.write(f"Warning: No valid data points in {os.path.basename(f_path)}, skipping.")
            continue
        
        true_bw = true_bw[valid_indices]
        pred_bw = pred_bw[valid_indices]

        mse = (true_bw - pred_bw) ** 2
        accuracy = np.maximum(0, 1 - np.abs(pred_bw - true_bw) / true_bw)
        overestimation = np.maximum(0, (pred_bw - true_bw) / true_bw)

        every_call_mse.append(np.mean(mse))
        every_call_accuracy.append(np.mean(accuracy))
        every_call_over.append(np.mean(overestimation))

    final_mse = np.mean(every_call_mse) if every_call_mse else 0
    final_accuracy = np.mean(every_call_accuracy) if every_call_accuracy else 0
    final_over = np.mean(every_call_over) if every_call_over else 0
    
    return final_mse, final_accuracy, final_over

def evaluate_policy(actor, eval_dataset_paths: list, device: str, batch_size: int = 512):
    actor.eval() # 设置为评估模式
    every_call_mse = []
    every_call_accuracy = []
    every_call_over = []

    progress_bar = tqdm(eval_dataset_paths, desc="Initializing policy evaluation...", position=0, leave=True)

    for f_path in progress_bar:
        progress_bar.set_description(f"Processing {os.path.basename(f_path)}")

        with open(f_path, 'rb') as f:
            call_data = pickle.load(f)

        # 1. 加载并预处理所有观测值
        observations = np.asarray(call_data["observations"], dtype=np.float32)
        true_capacity = np.asarray(call_data["true_capacities"], dtype=np.float32)
        
        # 归一化
        observations_norm = observations * NORMAL_VECTOR
        
        all_predictions = []
        num_obs = len(observations_norm)

        # 2. 向量化推理：一次性获得所有动作预测
        # 分批次进行推理
        with torch.no_grad():
            for i in range(0, num_obs, batch_size):
                batch_obs = observations_norm[i:i+batch_size]
                obs_tensor = torch.tensor(batch_obs, device=device, dtype=torch.float32)
                
                actions_tensor = actor.sample(obs_tensor)
                all_predictions.append(actions_tensor.cpu().numpy())

        # 合并所有批次的结果
        model_predictions = np.concatenate(all_predictions).flatten() / 1e6  # 假设动作单位是 bps
        true_bw = true_capacity / 1e6

        # 3. 向量化计算指标
        valid_indices = ~np.isnan(true_bw) & ~np.isnan(model_predictions) & (true_bw > 0)
        if not np.any(valid_indices):
            tqdm.write(f"Warning: No valid data points in {os.path.basename(f_path)}, skipping.")
            continue

        true_bw = true_bw[valid_indices]
        pred_bw = model_predictions[valid_indices]

        mse = (true_bw - pred_bw) ** 2
        accuracy = np.maximum(0, 1 - np.abs(pred_bw - true_bw) / true_bw)
        overestimation = np.maximum(0, (pred_bw - true_bw) / true_bw)

        every_call_mse.append(np.mean(mse))
        every_call_accuracy.append(np.mean(accuracy))
        every_call_over.append(np.mean(overestimation))

    actor.train() # 恢复为训练模式

    final_mse = np.mean(every_call_mse) if every_call_mse else 0
    final_accuracy = np.mean(every_call_accuracy) if every_call_accuracy else 0
    final_over = np.mean(every_call_over) if every_call_over else 0
    
    return final_mse, final_accuracy, final_over

if __name__ == "__main__":
    # --- 示例：如何调用新的 evaluate_policy ---
    
    # # 1. 加载你的模型 (这里只是一个示例结构)
    # from diffusion.ql_diffusion_e2e import Diffusion_QL
    # STATE_DIM = 66
    # ACTION_DIM = 1
    # MAX_ACTION = 20.0
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # agent = Diffusion_QL(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_ACTION, device=DEVICE, discount=0.99, tau=0.005)
    # agent.load_model('results_exp/test', id='10')
    
    # # 2. 定义评估数据集
    # eval_paths = [
    #     '/home/min414/data2/extra_storage/BoB_3.pickle',
    # ]

    # # 3. 调用评估函数
    # mse, accuracy, over = evaluate_policy(agent.actor, eval_paths, DEVICE)
    # print(f"\nPolicy Evaluation Results -- MSE: {mse:.4f}, Accuracy: {accuracy:.4f}, Over-Provision: {over:.4f}")

    # # --- 运行 evaluate_trace ---
    # print("--- Running Trace Evaluation ---")
    # mse_trace, acc_trace, over_trace = evaluate_trace(eval_paths)
    # print(f"\nTrace Evaluation Results -- MSE: {mse_trace:.4f}, Accuracy: {acc_trace:.4f}, Over-Provision: {over_trace:.4f}")

    
    # --- 示例：如何调用新的 evaluate_trace ---
    eval_paths = [
        '/home/min414/data2/extra_storage/BoB_3.pickle',
    ]
    mse_trace, acc_trace, over_trace = evaluate_trace(eval_paths)
    print(f"\nTrace Evaluation Results -- MSE: {mse_trace:.4f}, Accuracy: {acc_trace:.4f}, Over-Provision: {over_trace:.4f}")
    