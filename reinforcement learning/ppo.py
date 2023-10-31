from collections import namedtuple
from typing import Tuple, Optional
from torch import Tensor
from torch.distributions import MultivariateNormal, Categorical
from utils import get_device, plot_rewards

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym


class ReplayBuffer:
    """Rollout buffer for PPO
    
    """
    Transition = namedtuple('Transition',
                            ('state', 'action', 'logprobs', 'reward', 'value', 'next_state'))

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
                 env: gym.Env,
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
        self.env = env
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

    def evaluate(self, state, action):
        _, _ = self.actor_critic.act(state)
        value = self.actor_critic.evaluate(state)
        action_logprob = self.actor_critic.distribution.log_prob(action)
        entropy = self.actor_critic.entropy()

        return action_logprob, value, entropy

    def step(self, obs: Tensor) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor, bool]:
        """rollout one step"""
        action, logprob, value = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
        if done:
            next_obs = None
        else:
            next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

        self.replay_buffer.push(obs, action, logprob, reward, value, next_obs)
        return next_obs, action, reward, value, done

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs.unsqueeze(0)

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

    def update(self):
        sample = self.replay_buffer.sample()
        batch = self.replay_buffer.Transition(*zip(*sample))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        action_logprobs_batch = torch.cat(batch.logprobs)
        reward_batch = torch.cat(batch.reward)
        value_batch = torch.cat(batch.value)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat([torch.tensor(False, device=self.device)
                                if ns is None else torch.tensor(True, device=self.device) for ns in batch.next_state])

        advantages = self.compute_returns(state_batch[:, -1], value_batch, reward_batch, done_batch)

        for i in range(self.num_learning_epochs):
            action_logprobs, values, entropy = self.evaluate(state_batch, action_batch)

            # ratio = (pi_theta / pi_theta_old)
            ratios = torch.exp(action_logprobs - action_logprobs_batch.detach())

            # surrogate loss
            surrogate = -ratios * advantages
            surrogate_clipped = -torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # value loss
            value_loss = self.value_loss_coef * F.mse_loss(values, returns).mean()

            # entropy loss
            entropy_loss = self.entropy_coef * entropy.mean()

            loss = surrogate_loss + value_loss - entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def learn(self, num_episodes):
        """training process"""
        # rollout
        episode_durations = []
        for i in range(num_episodes):
            done = False
            episode_rewards = 0
            state = self.reset()
            duration = 0
            while not done:
                next_state, action, reward, value, done = self.step(state)
                episode_rewards += reward.item()
                state = next_state
                self.update()
                duration += 1

            episode_durations.append(duration + 1)
            print("step: {}, rew: {}, duration: {}".format(i + 1, episode_rewards, duration + 1))
            plot_rewards(show_result=False, episode_durations=episode_durations)

        plot_rewards(show_result=True, episode_durations=episode_durations)


