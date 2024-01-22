import os
import sys
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, 
                 num_states: int, 
                 num_actions: int,
                 hidden_dim: int,
                 init_noise_std: float = 1.0,
                 continuous_action: bool = False,
                 device: torch.device = "cpu",
                 ) -> None:
        super(ActorCritic, self).__init__()
        self.continuous_action = continuous_action
        if continuous_action:
            self.actor = nn.Sequential(
                nn.Linear(num_states, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_actions),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(num_states, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_actions),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.to(device)

    def act(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        if self.continuous_action:
            action_mean = self.actor(obs)
            self.distribution = MultivariateNormal(action_mean, action_mean * 0. + self.std)
        else:
            action_probs = self.actor(obs)
            self.distribution = Categorical(action_probs)

        action = self.distribution.sample()
        action_logprobs = self.distribution.log_prob(action)
        return action, action_logprobs

    def evaluate(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, ...]:
        action_mean = self.actor(obs)
        if self.continuous_action:
            dist = MultivariateNormal(action_mean)
        else:
            dist = Categorical(action_mean)
        value = self.critic(obs)
        action_logprob = dist.log_prob(action)

        return action_logprob, value, dist.logits
