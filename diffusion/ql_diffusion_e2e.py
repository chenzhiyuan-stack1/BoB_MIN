# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion.diffusion_id_e2e import Diffusion
from diffusion.model import MLP_GRU_v2
from diffusion.utils.helpers import EMA

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class V_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(V_Critic, self).__init__()
        self.v_model = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
    def forward(self, state):
        return self.v_model(state)

class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device='cpu',
                 discount=0.99,
                 tau=0.005,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=10,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=1e-5,
                 lr_decay=True,
                 lr_maxt=4000,
                 grad_norm=1.0,
                 args=None,
                 ):

        self.model = MLP_GRU_v2(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps, eta=eta, args=args,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-6)
        
        self.v_critic = V_Critic(state_dim).to(device)
        self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=5e-6)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100):
        metric = {
            'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'v_loss': [],
            'target_q': [], 'next_v': [], 'v': [], 'q1': [], 'q2': [], 'qs_for_policy': []
        }
        for _ in range(iterations):
            # Sample replay buffer / batch
            # action (Mbps)
            state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)
            
            """ Update Critic """
            with torch.no_grad():
                target_q = self.critic_target.q_min(state, action)  # next action from target actor
                # target_q = self.critic.q_min(state, action)
                next_v = self.v_critic(next_state)
            v = self.v_critic(state)
            adv = target_q - v
            v_loss = asymmetric_l2_loss(adv, 0.7)
            self.v_critic_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.v_critic_optimizer.step()
            
            """Update Critic Q functions"""
            targets = reward + self.discount * next_v
            targets = torch.clamp(targets, -200.0, 200.0)
            q1, q2 = self.critic(state, action)
            critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()


            """ Policy Training """
            with torch.no_grad():
                # adv = self.critic_target.q_min(state, action) - self.v_critic(state)
                adv = self.critic.q_min(state, action) - self.v_critic(state)
            bc_loss = self.actor.loss(action, state, adv)
            actor_loss = bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            # Update critic target network
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1
            
            """ Log """
            metric['ql_loss'].append(critic_loss.item())
            metric['v_loss'].append(v_loss.item())
            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['target_q'].append(target_q.mean().item())
            metric['next_v'].append(next_v.mean().item())
            metric['v'].append(v.mean().item())
            metric['q1'].append(q1.mean().item())
            metric['q2'].append(q2.mean().item())
            metric['qs_for_policy'].append(qs.mean().item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()

        return metric

    def train_debug(self, replay_buffer, iterations, batch_size=100):
        metric = {
            'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'v_loss': [],
            'target_q': [], 'next_v': [], 'v': [], 'q1': [], 'q2': [], 'qs_for_policy': []
        }
        for i in range(iterations):
            # Sample replay buffer / batch
            state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            reward = reward.to(self.device)
            not_done = not_done.to(self.device)

            # ======================= DEBUGGING BLOCK 1: Check Inputs =======================
            if i == 0: # 只在第一次迭代时检查，避免刷屏
                print("--- Iteration 0: Input Check ---")
                if torch.isnan(state).any() or torch.isinf(state).any():
                    print("!!! FATAL: NaN/inf found in 'state' input!")
                if torch.isnan(action).any() or torch.isinf(action).any():
                    print("!!! FATAL: NaN/inf found in 'action' input!")
                if torch.isnan(reward).any() or torch.isinf(reward).any():
                    print("!!! FATAL: NaN/inf found in 'reward' input!")
                print(f"Reward stats: mean={reward.mean().item():.4f}, min={reward.min().item():.4f}, max={reward.max().item():.4f}")
                print("--- End Input Check ---")
            # ===============================================================================

            """ Update Critic """
            with torch.no_grad():
                target_q = self.critic.q_min(state, action)
                next_v = self.v_critic(next_state)
            
            v = self.v_critic(state)
            adv = target_q - v
            v_loss = asymmetric_l2_loss(adv, 0.7)

            # ======================= DEBUGGING BLOCK 2: V-Loss Calculation =======================
            if torch.isnan(v_loss).any():
                print("\n!!! V-Loss is NaN. Checking components:")
                print(f"target_q has NaN: {torch.isnan(target_q).any()}, has inf: {torch.isinf(target_q).any()}")
                print(f"next_v has NaN: {torch.isnan(next_v).any()}, has inf: {torch.isinf(next_v).any()}")
                print(f"v has NaN: {torch.isnan(v).any()}, has inf: {torch.isinf(v).any()}")
                print(f"adv has NaN: {torch.isnan(adv).any()}, has inf: {torch.isinf(adv).any()}")
                # 提前终止，防止错误蔓延
                raise RuntimeError("NaN detected in V-Loss calculation. Aborting.")
            # =====================================================================================

            self.v_critic_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.v_critic_optimizer.step()
            
            """Update Critic Q functions"""
            with torch.no_grad():
                targets = reward + self.discount * next_v
            
            q1, q2 = self.critic(state, action)
            critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)

            # ======================= DEBUGGING BLOCK 3: Critic-Loss Calculation =======================
            if torch.isnan(critic_loss).any():
                print("\n!!! Critic-Loss is NaN. Checking components:")
                print(f"targets has NaN: {torch.isnan(targets).any()}, has inf: {torch.isinf(targets).any()}")
                print(f"q1 has NaN: {torch.isnan(q1).any()}, has inf: {torch.isinf(q1).any()}")
                print(f"q2 has NaN: {torch.isnan(q2).any()}, has inf: {torch.isinf(q2).any()}")
                raise RuntimeError("NaN detected in Critic-Loss calculation. Aborting.")
            # ==========================================================================================

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()


            """ Policy Training """
            with torch.no_grad():
                qs = self.critic_target.q_min(state, action)
            
            bc_loss = self.actor.loss(action, state, qs)
            actor_loss = bc_loss

            # ======================= DEBUGGING BLOCK 4: Actor-Loss Calculation =======================
            if torch.isnan(actor_loss).any():
                print("\n!!! Actor-Loss is NaN. Checking components:")
                print(f"qs has NaN: {torch.isnan(qs).any()}, has inf: {torch.isinf(qs).any()}")
                # bc_loss 内部复杂，先确认输入 qs
                raise RuntimeError("NaN detected in Actor-Loss calculation. Aborting.")
            # =========================================================================================

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            metric['ql_loss'].append(critic_loss.item())
            metric['v_loss'].append(v_loss.item())
            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['target_q'].append(target_q.mean().item())
            metric['next_v'].append(next_v.mean().item())
            metric['v'].append(v.mean().item())
            metric['q1'].append(q1.mean().item())
            metric['q2'].append(q2.mean().item())
            metric['qs_for_policy'].append(qs.mean().item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.to("cpu").reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=10, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic.q_min(state_rpt, action / 1e6).flatten()
            q_value = torch.clamp(q_value, min=0)
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
            torch.save(self.v_critic.state_dict(), f'{dir}/v_critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')
            torch.save(self.v_critic.state_dict(), f'{dir}/v_critic.pth')
            
    def save_newest_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/MDQL.pth')

    def load_newest_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/MDQL.pth'))

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
            self.v_critic.load_state_dict(torch.load(f'{dir}/v_critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
            self.v_critic.load_state_dict(torch.load(f'{dir}/v_critic.pth'))
        self.critic_target = copy.deepcopy(self.critic)
