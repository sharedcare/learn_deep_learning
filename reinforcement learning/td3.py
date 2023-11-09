from collections import namedtuple, deque
from typing import Tuple, Optional
from torch import Tensor

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

    def forward(self, state: Tensor) -> Tensor:
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
    def __init__(self,
                 env: gym.Env,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 batch_size: int,
                 lr: float,
                 mem_capacity: int,
                 tau: float,
                 eps: float,
                 gamma: float,
                 noise_clip: float,
                 action_max: int,
                 num_updates: int,
                 policy_delay: int,
                 device: torch.device = "cpu") -> None:
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.device = device

        self.actor = Actor(num_states, num_actions, hidden_dim, action_max)
        self.target_actor = Actor(num_states, num_actions, hidden_dim, action_max)
        self.critic = Critic(num_states, hidden_dim)
        self.target_critic = Critic(num_states, hidden_dim)
        self.replay_buffer = ReplayBuffer(mem_capacity)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # TD3 parameters
        self.tau = tau
        self.eps = eps
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.action_max = action_max
        self.num_updates = num_updates
        self.policy_delay = policy_delay

    def act(self, state: Tensor) -> Tensor:
        action_mean = self.actor(state)
        return torch.clamp(action_mean + self.eps * torch.randn_like(action_mean), -self.action_max, self.action_max)

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

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs.unsqueeze(0)

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return
        for i in self.num_updates:
            # randomly sample from replay buffer
            sample = self.replay_buffer.sample(self.batch_size)
            batch = self.replay_buffer.Transition(*zip(*sample))

            next_state_batch = torch.cat([torch.zeros(1, self.num_states, device=self.device)
                                          if ns is None else ns for ns in batch.next_state])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            done_batch = torch.tensor([0. if ns is None else 1. for ns in batch.next_state], device=self.device)

            with torch.no_grad():
                # compute target actions
                next_action_batch = torch.clamp(self.target_actor(next_state_batch) +
                                                torch.clamp(self.eps, -self.noise_clip, self.noise_clip),
                                                -self.action_max, self.action_max)

                # calculate targets
                target_q1, target_q2 = self.target_critic(next_state_batch, next_action_batch)
                target_q = reward_batch + self.gamma * (1 - done_batch) * torch.min(target_q1, target_q2)

            # update Q-functions
            q1, q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if i % self.policy_delay == 0:
                # update policy
                action_loss = self.critic.critic1(state_batch, self.actor(state_batch)).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                self.actor_optimizer.step()

                # update target networks
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * target_param + (1 - self.tau) * param)
                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(self.tau * target_param + (1 - self.tau) * param)

    def learn(self, num_episodes: int) -> None:
        """training process"""
        episode_durations = []
        for i in range(num_episodes):
            done = False
            episode_rewards = 0
            state = self.reset()
            duration = 0
            while not done:
                next_state, action, reward, done = self.step(state)
                episode_rewards += reward.item()
                state = next_state
                self.update()
                duration += 1

            episode_durations.append(duration + 1)
            print("step: {}, rew: {}, duration: {}".format(i + 1, episode_rewards, duration + 1))
            plot_rewards(show_result=False, episode_durations=episode_durations)

        plot_rewards(show_result=True, episode_durations=episode_durations)

