from collections import namedtuple, deque
from torch.distributions import MultivariateNormal, Categorical

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


class ReplayBuffer:
    """Rollout buffer for PPO
    
    """
    Transition = namedtuple('Transition',
                            ('state', 'action', 'logprobs', 'reward', 'next_state'))

    def __init__(self, device="cpu") -> None:
        self.mem = []
        self.count = 0

    def push(self, *transition):
        """ save transition to buffer """
        self.mem.append(self.Transition(*transition))
        self.count += 1

    def sample(self, batch_size=None):
        """ sample transitions from memory buffer """
        if batch_size is None:
            return self.Transition(*zip(*self.mem))
        else:
            return self.Transition(*zip(*random.sample(self.mem, batch_size)))

    def __len__(self) -> int:
        return len(self.mem)


class ActorCritic(nn.Module):
    def __init__(self,
                 num_states,
                 num_actions,
                 hidden_dim,
                 init_noise_std=1.0,
                 continuous_action=False
                 ):
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

        self.distribution = None
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

    def forward(self):
        raise NotImplementedError

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs):
        if self.continuous_action:
            action_mean = self.actor(obs)
            self.distribution = MultivariateNormal(action_mean, action_mean * 0. + self.std)
        else:
            action_probs = self.actor(obs)
            self.distribution = Categorical(action_probs)

        action = self.distribution.sample()
        action_logprobs = self.distribution.log_prob(action)
        return action, action_logprobs

    def evaluate(self, obs):
        value = self.critic(obs)
        return value


class PPO:
    def __init__(self,
                 num_states,
                 num_actions,
                 hidden_dim,
                 learning_rate,
                 continuous_action,
                 clip_param,
                 value_clip_param,
                 num_learning_epochs,
                 value_loss_coef,
                 entropy_coef,
                 gamma,
                 lam,
                 max_grad_norm,
                 device="cpu"):
        self.device = device
        self.continuous_action = continuous_action
        self.replay_buffer = ReplayBuffer()
        self.actor_critic = ActorCritic(num_states, num_actions, hidden_dim, continuous_action)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = value_clip_param

    def act(self, state):
        action, action_logprobs = self.actor_critic.act(state)
        value = self.actor_critic.evaluate(state)
        # self.replay_buffer.push(state, action, action_logprobs)
        if self.continuous_action:
            return action.detach(), action_logprobs, value
        else:
            return action.item(), action_logprobs, value

    def compute_returns(self, last_state, values, rewards, dones):
        """ GAE """
        last_values = self.actor_critic.evaluate(last_state)
        advantage = 0
        advantages = torch.zeros_like(rewards)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * last_values - values[step]     # td error
            advantage = delta + self.gamma * self.lam * advantage
            advantages[step] = advantage

        return advantages

