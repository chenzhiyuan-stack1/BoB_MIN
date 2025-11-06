# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from typing import Dict, List

TensorBatch = List[torch.Tensor]

class ReplayBuffer:
    """
    A replay buffer for off-policy RL agents.
    Supports initial dataset loading and online, incremental additions with a circular buffer logic.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        
        print(f"ReplayBuffer initialized with size {self._buffer_size} on device '{self._device}'")

    @property
    def size(self) -> int:
        return self._size

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]):
        """
        Loads a dataset, completely overwriting the buffer's contents.
        This is typically used for initializing the buffer in offline training.
        """
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(f"Replay buffer (size: {self._buffer_size}) is smaller than the dataset you are trying to load (size: {n_transitions})!")
        
        # Clear buffer before loading
        self._pointer = 0
        self._size = 0

        # Use add_transitions to load the data, which handles the logic correctly
        self.add_transitions(data)
        print(f"Dataset loaded. Buffer size: {self._size}")

    def add_transitions(self, data: Dict[str, np.ndarray]):
        """
        Adds new transitions to the buffer using a circular logic.
        If the buffer is full, it overwrites the oldest data.
        This is the correct method for online learning.
        """
        n_transitions = data["observations"].shape[0]
        
        # Convert all numpy arrays to tensors once
        states_tensor = self._to_tensor(data["observations"])
        actions_tensor = self._to_tensor(data["actions"])
        rewards_tensor = self._to_tensor(data["rewards"][..., None])
        next_states_tensor = self._to_tensor(data["next_observations"])
        dones_tensor = self._to_tensor(data["terminals"][..., None])

        # Check if the new data will wrap around the buffer
        remaining_space = self._buffer_size - self._pointer
        if n_transitions > remaining_space:
            # First part: fill the rest of the buffer
            part1_len = remaining_space
            self._states[self._pointer : self._pointer + part1_len] = states_tensor[:part1_len]
            self._actions[self._pointer : self._pointer + part1_len] = actions_tensor[:part1_len]
            self._rewards[self._pointer : self._pointer + part1_len] = rewards_tensor[:part1_len]
            self._next_states[self._pointer : self._pointer + part1_len] = next_states_tensor[:part1_len]
            self._dones[self._pointer : self._pointer + part1_len] = dones_tensor[:part1_len]

            # Second part: wrap around and fill from the beginning
            part2_len = n_transitions - part1_len
            self._states[0:part2_len] = states_tensor[part1_len:]
            self._actions[0:part2_len] = actions_tensor[part1_len:]
            self._rewards[0:part2_len] = rewards_tensor[part1_len:]
            self._next_states[0:part2_len] = next_states_tensor[part1_len:]
            self._dones[0:part2_len] = dones_tensor[part1_len:]
            
            self._pointer = part2_len
        else:
            # If it fits, just add it sequentially
            self._states[self._pointer : self._pointer + n_transitions] = states_tensor
            self._actions[self._pointer : self._pointer + n_transitions] = actions_tensor
            self._rewards[self._pointer : self._pointer + n_transitions] = rewards_tensor
            self._next_states[self._pointer : self._pointer + n_transitions] = next_states_tensor
            self._dones[self._pointer : self._pointer + n_transitions] = dones_tensor
            
            self._pointer += n_transitions

        # Update the total size, capped by the buffer size
        self._size = min(self._size + n_transitions, self._buffer_size)
        print(f"Transitions added. New buffer size: {self._size}, Pointer at: {self._pointer}")

    def sample(self, batch_size: int) -> TensorBatch:
        """
        Samples a batch of transitions from the buffer.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        ]

# --- Legacy Data_Sampler Class (kept for compatibility) ---

class Data_Sampler(object):
    def __init__(self, data, device, reward_tune='no'):
        
        self.state = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.reward = reward

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device)
        )


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
    reward /= (rt_max - rt_min)
    reward *= 1000.
    return reward
