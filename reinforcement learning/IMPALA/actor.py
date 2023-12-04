import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

from torch import Tensor


class Actor(nn.Module):
    def __init__(self, 
                 num_states: int, 
                 num_actions: int,
                 hidden_dim: int,
                 init_noise_std: float = 1.0,
                 continuous_action: bool = False,
                 device: torch.device = "cpu",
                 ) -> None:
        super(Actor, self).__init__()
        if continuous_action:
            self.actor = nn.Sequential(
                nn.Linear(num_states, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_actions),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(num_states, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_actions),
                nn.Softmax(),
            )
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.to(device)

    def forward(self, obs: Tensor) -> Tensor:
        action_mean = self.actor(obs)
        return action_mean