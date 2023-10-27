import torch
import torch.nn as nn
import numpy as np
import gym


class RolloutBuffer:
    """Rollout buffer for PPO
    
    """
    def __init__(self, batch_size, num_states, num_actions, device="cpu") -> None:
        self.states = torch.zeros(batch_size, num_states, device=device)
        self.actions = torch.zeros(batch_size, num_actions, device=device)
        self.logprobs = torch.zeros(batch_size, num_actions, device=device)
        self.next_states = torch.zeros_like(self.states)
        self.rewards = torch.zeros(batch_size, 1, device=device)
        self.dones = torch.zeros(batch_size, 1, device=device)

        self.count = 0

    def store_transitions(self, *args):
        self.states[self.count] = args[0]
        self.actions[self.count] = args[1]
        self.logprobs[self.count] = args[2]
        self.next_states[self.count] = args[3]
        self.rewards[self.count] = args[4]
        self.dones[self.count] = args[5]

        self.count += 1

