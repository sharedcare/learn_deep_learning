import sys
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

sys.path.append('../')

from utils import get_device


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
                 env: gym.Env,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 eps_start: float,
                 eps_end: float,
                 eps_decay: int,
                 gamma: float,
                 mem_capacity: int,
                 batch_size: int,
                 device: torch.device = "cpu") -> None:
        super(DQN, self).__init__()
        self.env = env
        self.policy_net = Net(num_states, num_actions, hidden_dim)
        self.target_net = Net(num_states, num_actions, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.device = device

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(mem_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

        self.steps_done = 0

    def act(self, obs) -> Tensor:
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        if random.random() > epsilon:
            with torch.no_grad():
                action_value = self.policy_net(obs)
                action = torch.max(action_value, 1)[1]
                return action.view(1, 1)
        else:
            return torch.tensor([self.env.action_space.sample()])

    def step(self, obs) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
        action = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
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
        # randomly sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*sample))

        non_terminal_next_state_batch = torch.cat([ns for ns in batch.next_state if ns is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a): compute Q(s) and select taken actions as state action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s_{t+1}): \max_a Q(s_{t+1}, a)
        next_state_values = torch.max(self.target_net(non_terminal_next_state_batch), 1)[0]

        # Expected Q(s,a): r + \gamma V(s_{t+1}) using Bellman equation
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # TD error: Q(s_t, a) - Expected Q(s, a)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, num_episodes):
        for i in range(num_episodes):
            done = False
            episode_rewards = 0
            state = self.reset()
            while not done:
                next_state, action, reward, done = self.step(state)
                episode_rewards += reward
                self.update()

            print("step: {}, rew: {}".format(i + 1, episode_rewards))


if __name__ == "__main__":
    EPSILON_START = 0.9         # start epsilon for greedy policy
    EPSILON_END = 0.05          # minimum epsilon
    EPSILON_DECAY = 1000        # decay rate from epsilon
    BATCH_SIZE = 32             # sample batch size
    GAMMA = 0.99                # reward discount
    LR = 1e-3                   # learning rate
    MEMORY_CAPACITY = 10000     # replay buffer memory capacity
    NUM_EPISODES = 500          # number of episodes for sampling and training
    HIDDEN_DIM = 128            # hidden dimension size for Q-network

    gym_env = gym.make("CartPole-v1").unwrapped
    n_states = gym_env.observation_space.shape[0]
    n_actions = gym_env.action_space.n

    run_device = get_device()

    dqn = DQN(env=gym_env,
              num_states=n_states,
              num_actions=n_actions,
              hidden_dim=HIDDEN_DIM,
              eps_start=EPSILON_START,
              eps_end=EPSILON_END,
              eps_decay=EPSILON_DECAY,
              gamma=GAMMA,
              mem_capacity=MEMORY_CAPACITY,
              batch_size=BATCH_SIZE,
              device=run_device,
              )

    dqn.learn(NUM_EPISODES)
