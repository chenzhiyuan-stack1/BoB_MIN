# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.utils.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
from diffusion.utils.utils import Progress, Silent

class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True, eta=1, args=None):
        super(Diffusion, self).__init__()  
        
        # Q and BC
        self.eta = eta
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas) # \beta_t
        self.register_buffer('alphas_cumprod', alphas_cumprod) # \prod_{i=0}^{t-1} (1 - \beta_i)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev) # \prod_{i=0}^{t-2} (1 - \beta_i)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)) # \sqrt{\prod_{i=0}^{t-1} (1 - \beta_i)}
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod)) # \sqrt{1 - \prod_{i=0}^{t-1} (1 - \beta_i)}
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod)) # \log(1 - \prod_{i=0}^{t-1} (1 - \beta_i))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod)) # \sqrt{1 / \prod_{i=0}^{t-1} (1 - \beta_i)}
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # \sqrt{1 / \prod_{i=0}^{t-1} (1 - \beta_i) - 1}

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # \beta_t * (1 - \prod_{i=0}^{t-2} (1 - \beta_i)) / (1 - \prod_{i=0}^{t-1} (1 - \beta_i))
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=(self.model(x / 1e6, t, s) * 1e6))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action * 1e6, self.max_action * 1e6)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action * 1e6, self.max_action * 1e6)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, adv, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long() # sample t
        beta_t = extract(self.betas, t, (batch_size,)).unsqueeze(1)
        alpha_t = 1.01 - beta_t
        alpha_cumprod_t_1 = extract(self.alphas_cumprod_prev, t, (batch_size,)).unsqueeze(1)
        
        exp_weight = torch.exp((1 / self.eta) * adv).clamp(max=10000)
        weights = beta_t / 2. / alpha_t / (1.01 - alpha_cumprod_t_1) * exp_weight
        loss = self.p_losses(x, state, t, weights)
        return loss
    
    def _p_mean_variance_with_model(self, x, t, s, policy_model):
        """
        复用 p_mean_variance，但可指定使用的 policy_model（当前或旧策略）。
        """
        # model(x/1e6) 与原逻辑一致
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=(policy_model(x / 1e6, t, s) * 1e6)
        )

        if self.clip_denoised:
            x_recon = x_recon.clamp(-self.max_action * 1e6, self.max_action * 1e6)

        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_recon +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, posterior_variance, posterior_log_variance_clipped

    def _log_prob_last_step(self, policy_model, x, state):
        """
        用 diffusion 最后一步（t=0）的高斯分布对给定 action x 计算 log_prob。
        """
        b = x.shape[0]
        t = torch.zeros(b, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance = self._p_mean_variance_with_model(x, t, state, policy_model)
        std = (0.5 * model_log_variance).exp()  # std = exp(log_var/2)
        # 高斯对角协方差，沿动作维度求和
        log_prob = (-0.5 * (((x - model_mean) / std)**2 + 2 * torch.log(std) + np.log(2 * np.pi))).sum(dim=-1)
        return log_prob

    def ppo_loss(self, x, state, adv, weights=1.0, clip_ratio=0.2):
        """
        x: 动作（policy 输出或采样得到的 action）
        state: 状态
        adv: 优势
        weights: 额外样本权重
        clip_ratio: PPO epsilon
        """
        # 当前策略 log_prob（需要梯度）
        logp_new = self._log_prob_last_step(self.model, x, state)
        # 旧策略 log_prob（无梯度）
        with torch.no_grad():
            logp_old = self._log_prob_last_step(self.old_model, x, state)

        ratio = torch.exp(logp_new - logp_old)
        # PPO clip
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        loss_per_sample = -torch.min(unclipped, clipped)

        if isinstance(weights, torch.Tensor):
            loss_per_sample = loss_per_sample * weights

        return loss_per_sample.mean()

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
