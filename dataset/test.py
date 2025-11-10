import pickle
import torch
import numpy as np
from diffusion.ql_diffusion_e2e import Diffusion_QL
from diffusion.norm_vector import NORMAL_VECTOR

# --- 1. 初始化 Agent 并加载模型 ---
agent = Diffusion_QL(
        state_dim=66,
        action_dim=1,
        max_action=20,
        device='cpu' # 明确指定设备
    )
# 假设模型文件在 'results_exp/your_exp_name/actor_240.pth' 等
# 请根据实际路径修改
agent.load_model("dataset/ckpt", '240')
print("Agent initialized. Make sure to load the correct model weights.")


# --- 2. 加载数据 ---
pickle_path = '/home/min414/data2/extra_storage/BoB_5.pickle'
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# --- 3. 准备单个输入样本 ---
# 从Numpy数组中获取数据
obs_np = data['observations'][2200] * NORMAL_VECTOR
actions_np = data['actions'][2200] / 1e6

# --- 核心改动：将Numpy数组转换为PyTorch张量，并添加批次维度 ---
# obs: 从 (66,) 变为 (1, 66)
obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)
# actions: 从 (1,) 变为 (1, 1)
actions_tensor = torch.from_numpy(actions_np).float().unsqueeze(0)


# --- 4. 将模型设置为评估模式 ---
agent.actor.eval()
agent.critic.eval()
agent.v_critic.eval()

print("Actor eval mode:", not agent.actor.training)
print("Critic eval mode:", not agent.critic.training)
print("V_Critic eval mode:", not agent.v_critic.training)


# --- 5. 使用正确维度的张量进行推理 ---
print("\nRunning inference...")
with torch.no_grad(): # 在推理时使用 no_grad
    # sample() 内部处理了设备，所以不需要 .to(device)
    actor_output = agent.actor.sample(obs_tensor)
    print("Actor sample output shape:", actor_output.shape)
    print("Actor sample output:", actor_output.numpy().flatten()[0])

    # critic 和 v_critic 需要手动将张量移动到设备
    q1_output, q2_output = agent.critic(obs_tensor, actions_tensor)
    print("Critic q1 output shape:", q1_output.shape)
    print("Critic q2 output shape:", q2_output.shape)
    print("Critic q1 output:", q1_output)
    print("Critic q2 output:", q2_output)

    # v_critic 的 forward 方法是 v_model，而不是 v
    v_output = agent.v_critic(obs_tensor)
    print("V_Critic output shape:", v_output.shape)
    print("V_Critic output:", v_output)

print("\nInference successful.")
