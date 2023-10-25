import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from torch import Tensor
from collections import deque, namedtuple
from typing import List, Tuple, Optional


class ReplayBuffer(object):
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def __init__(self, capacity) -> None:
        self.mem = deque([], maxlen=capacity)

    def push(self, *transition) -> None:
        """ save transition to buffer """
        self.mem.append(self.Transition(*transition))

    def sample(self, batch_size) -> list:
        """ randomly sample memory buffer """
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)


class Net(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim) -> None:
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x) -> Tensor:
        action_value = self.layers(x)
        return action_value


class DQN:
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 mem_capacity: int,
                 batch_size: int,
                 device: str = "cpu") -> None:
        super(DQN, self).__init__()
        self.env = gym.make("cartpole")
        self.policy_net = Net(num_states, num_actions, hidden_dim)
        self.target_net = Net(num_states, num_actions, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.device = device

        self.replay_buffer = ReplayBuffer(mem_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

        self.steps_done = 0

    def act(self, obs) -> Tensor:
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-1 * self.steps_done / eps_decay)
        if random.random() > epsilon:
            with torch.no_grad():
                action_value = self.policy_net(obs)
                action = torch.max(action_value, 1)[1]
                return action.view(1, 1)
        else:
            return torch.tensor([self.env.action_space.sample()])

    def step(self, obs) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
        action = self.act(obs)
        next_obs, reward, done, info = self.env.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        if done:
            next_obs = None
        else:
            next_obs = torch.tensor([next_obs], device=self.device)

        self.replay_buffer.push(obs, action, reward, next_obs)
        return next_obs, action, reward, done

    def reset(self) -> Tensor:
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs

    def update(self):
        if len(self.replay_buffer) < self.mem_capacity:
            return
        sample = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*sample))

        non_terminal_next_state_batch = torch.cat([ns for ns in batch.next_state if ns is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a): compute Q(s) and select taken actions as state action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s_{t+1}): \max_a Q(s_{t+1}, a)

    def learn(self, num_episodes):
        for i in range(num_episodes):
            done = False
            episode_rewards = 0
            state = self.reset()
            while not done:
                next_state, action, reward, done = self.step(state)
                episode_rewards += reward
