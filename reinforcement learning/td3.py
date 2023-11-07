from collections import namedtuple, deque
from typing import Tuple, Optional
from torch import Tensor
from torch.distributions import MultivariateNormal, Categorical

import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

sys.path.append('./')

from utils import get_device, plot_rewards


class ReplayBuffer(object):
    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state'))

    def __init__(self, capacity: int) -> None:
        self.mem = deque([], maxlen=capacity)

    def push(self, *transition: Tensor) -> None:
        """ save transition to buffer """
        self.mem.append(self.Transition(*transition))

    def sample(self, batch_size: int) -> list:
        """ randomly sample memory buffer """
        return random.sample(self.mem, batch_size)

    def __len__(self) -> int:
        return len(self.mem)


class Actor(nn.Module):
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 max_action,
                 device: torch.device = "cpu",
                 ) -> None:
        super(Actor, self).__init__()
        # TD3 is for contiunous action space only
        self.actor = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh()
        )

        self.max_action = max_action
        self.to(device)

    def forward(self, state: Tensor):
        action = self.actor(state)
        return action * self.max_action

    @property
    def entropy(self) -> Tensor:
        return self.distribution.entropy().sum(dim=-1)


class Critic(nn.Module):
    def __init__(self,
                 num_states: int,
                 hidden_dim: int,
                 device: torch.device = "cpu",
                 ) -> None:
        super(Critic, self).__init__()

        self.critic1 = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(device)
    
    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        sa = torch.cat([state, action])
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        return q1, q2


class TD3:
    def __init__(self) -> None:
        self.actor = Actor()
        self.target_actor = Actor()
        self.critic = Critic()
        self.target_critic = Critic()
        self.replay_buffer = ReplayBuffer()
        self.device = device

    def act(self, state: Tensor) -> Tensor:
        action = self.actor(state)
        return action

    def step(self, obs: Tensor) -> Tuple[Tensor, ...]:
        """rollout one step"""
        action = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
        if done:
            next_obs = None
        else:
            next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

        self.replay_buffer.push(obs, action, reward, next_obs)
        return next_obs, action, reward, done

    def update(self):
        pass

    def learn(self):
        pass

