import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym


class Net(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions = self.fc2(x)
        return actions

