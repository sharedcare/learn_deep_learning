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


class ReplayBuffer:
    def __init__(self, max_size, input_shape, num_actions, device: torch.device = "cpu"):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, input_shape), device=device)
        self.next_state_memory = torch.zeros((self.mem_size, input_shape), device=device)
        self.action_memory = torch.zeros((self.mem_size, num_actions), device=device)
        self.reward_memory = torch.zeros(self.mem_size, device=device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=device)
        self.device = device

    def store(self, state, action, reward, state_, done):
        """ save transition to buffer """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample(self, batch_size):
        """ randomly sample memory buffer """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = torch.randperm(max_mem, requires_grad=False, device=self.device)[:batch_size]

        states = self.state_memory[batch].detach()
        next_states = self.next_state_memory[batch].detach()
        actions = self.action_memory[batch].detach()
        rewards = self.reward_memory[batch].detach()
        dones = self.terminal_memory[batch].detach()

        return states, actions, rewards, next_states, dones


class Actor(nn.Module):
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 max_action: float,
                 device: torch.device = "cpu",
                 ) -> None:
        super(Actor, self).__init__()
        # TD3 is for contiunous action space only
        self.actor = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh()
        )

        self.max_action = max_action
        self.to(device)

    def forward(self, state: Tensor) -> Tensor:
        action = self.actor(state)
        return action

    @property
    def entropy(self) -> Tensor:
        return self.distribution.entropy().sum(dim=-1)


class Critic(nn.Module):
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 device: torch.device = "cpu",
                 ) -> None:
        super(Critic, self).__init__()

        self.critic1 = nn.Sequential(
            nn.Linear(num_states + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(num_states + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(device)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        return q1, q2

    def q1(self, state: Tensor, action: Tensor) -> Tensor:
        sa = torch.cat([state, action], dim=-1)
        q1 = self.critic1(sa)
        return q1


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
                 action_max: float,
                 action_min: float,
                 start_steps: int,
                 policy_delay: int,
                 device: torch.device = "cpu") -> None:
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.timestep = 0
        self.device = device

        self.actor = Actor(num_states, num_actions, hidden_dim, action_max, device=device)
        self.target_actor = Actor(num_states, num_actions, hidden_dim, action_max, device=device)
        self.critic = Critic(num_states, num_actions, hidden_dim, device=device)
        self.target_critic = Critic(num_states, num_actions, hidden_dim, device=device)
        self.replay_buffer = ReplayBuffer(mem_capacity, num_states, num_actions, device=device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # TD3 parameters
        self.tau = tau
        self.eps = eps
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.action_max = action_max
        self.action_min = action_min
        self.num_update = 0
        self.start_steps = start_steps
        self.policy_delay = policy_delay

    def save(self, path: str) -> None:
        """save model to designated path"""
        torch.save(self.actor.state_dict(), path + "_actor.pt")
        torch.save(self.critic.state_dict(), path + "_critic.pt")
        print("model saved at: {}".format(path))

    def load(self, path: str) -> None:
        """load network from path"""
        self.actor.load_state_dict(torch.load(path + "_actor.pt"))
        self.critic.load_state_dict(torch.load(path + "_critic.pt"))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        print("model loaded from: {}".format(path))

    def act(self, state: Tensor) -> Tensor:
        action_mean = self.actor(state)
        return torch.clamp(action_mean + self.eps * torch.randn_like(action_mean), self.action_min, self.action_max)

    def step(self, obs: Tensor) -> Tuple[Optional[Tensor], Tensor, Tensor, bool]:
        """rollout one step"""
        if self.timestep < self.start_steps:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            action = torch.tensor(action, device=self.device).unsqueeze(0)
        else:
            action = self.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action.detach().cpu().numpy().flatten())
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32).unsqueeze(0)

        self.replay_buffer.store(obs, action, reward, next_obs, done)
        return next_obs, action, reward, done

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        return obs.unsqueeze(0)

    def update(self) -> None:
        if self.replay_buffer.mem_cntr < self.batch_size:
            return
        self.num_update += 1
        # randomly sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = sample

        with torch.no_grad():
            # compute target actions
            next_action_batch = torch.clamp(self.target_actor(next_state_batch) +
                                            torch.clamp(torch.randn_like(action_batch) * self.eps,
                                                        -self.noise_clip, self.noise_clip),
                                            self.action_min, self.action_max)

            # calculate targets
            target_q1, target_q2 = self.target_critic(next_state_batch, next_action_batch)
            target_q = reward_batch + self.gamma * (1 - done_batch.long()) * torch.min(target_q1.squeeze(-1),
                                                                                       target_q2.squeeze(-1))

        # update Q-functions
        q1, q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(target_q.unsqueeze(-1), q1) + F.mse_loss(target_q.unsqueeze(-1), q2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.num_update % self.policy_delay == 0:
            # update policy
            action_loss = -self.critic.q1(state_batch, self.actor(state_batch)).mean()

            self.actor_optimizer.zero_grad()
            action_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)

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

            self.timestep += 1
            episode_durations.append(duration + 1)
            print("step: {}, rew: {}, duration: {}".format(i + 1, episode_rewards, duration + 1))
            plot_rewards(show_result=False, episode_durations=episode_durations)

        plot_rewards(show_result=True, episode_durations=episode_durations)


if __name__ == "__main__":
    BATCH_SIZE = 256                # sample batch size
    GAMMA = 0.99                    # reward discount
    TAU = 0.995                     # target network update rate
    LR = 5e-4                       # learning rate
    NUM_EPISODES = 1000             # number of episodes for sampling and training
    START_EPISODES = 50             # total episode steps for warmup random exploration
    MEMORY_CAPACITY = 1000000       # replay buffer memory capacity
    CLIP_PARAM = 0.5                # clip factor for td3 target policy noise
    HIDDEN_DIM = 256                # hidden dimension size for actor-critic network
    EPSILON = 0.2                   # std of Gaussian exploration noise
    POLICY_DELAY = 2                # frequency of delayed policy updates

    LOAD_MODEL_PATH = "saved_models/rl/td3.pt"
    SAVE_MODEL_PATH = "saved_models/rl/new_td3.pt"

    gym_env = gym.make("LunarLanderContinuous-v2").unwrapped
    n_states = gym_env.observation_space.shape[0]
    n_actions = gym_env.action_space.shape[0]
    max_action = float(gym_env.action_space.high[0])
    min_action = float(gym_env.action_space.low[0])

    if isinstance(gym_env.action_space, gym.spaces.Discrete):
        is_continuous_action = False
    else:
        is_continuous_action = True

    run_device = get_device()

    td3 = TD3(gym_env,
              n_states,
              n_actions,
              HIDDEN_DIM,
              BATCH_SIZE,
              LR,
              MEMORY_CAPACITY,
              TAU,
              EPSILON,
              GAMMA,
              CLIP_PARAM,
              max_action,
              min_action,
              START_EPISODES,
              POLICY_DELAY,
              device=run_device
              )

    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        td3.load(LOAD_MODEL_PATH)
    td3.learn(NUM_EPISODES)
    if SAVE_MODEL_PATH:
        td3.save(SAVE_MODEL_PATH)
