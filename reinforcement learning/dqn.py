import os.path
import sys
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import gym
from torch import Tensor
from collections import deque, namedtuple
from typing import List, Tuple, Optional

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


class Net(nn.Module):
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 device: torch.device = "cpu") -> None:
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.device = device
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
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
                 tau: float,
                 lr: float,
                 mem_capacity: int,
                 batch_size: int,
                 device: torch.device = "cpu") -> None:
        super(DQN, self).__init__()
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.policy_net = Net(num_states, num_actions, hidden_dim, device)
        self.target_net = Net(num_states, num_actions, hidden_dim, device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.device = device

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        self.replay_buffer = ReplayBuffer(mem_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.steps_done = 0

    def save(self, path: str) -> None:
        """save policy model to designated path"""
        torch.save(self.policy_net.state_dict(), path)
        print("Policy model saved at: {}".format(path))

    def load(self, path: str) -> None:
        """load network from path"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Policy model loaded from: {}".format(path))

    def act(self, obs: Tensor) -> Tensor:
        """choose action"""
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > epsilon:
            with torch.no_grad():
                action_value = self.policy_net(obs)
                action = torch.max(action_value, 1)[1]
                return action.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device)

    def step(self, obs: Tensor) -> Tuple[Optional[Tensor], Tensor, Tensor, bool]:
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

    def net_soft_update(self):
        """
        soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′

        """
        update_state_dict = copy.deepcopy(self.target_net.state_dict())
        for key in update_state_dict.keys():
            update_state_dict[key] = (self.target_net.state_dict()[key] * (1 - self.tau) +
                                      self.policy_net.state_dict()[key] * self.tau)
        self.target_net.load_state_dict(update_state_dict)

    def update(self) -> None:
        """update policy network"""
        if len(self.replay_buffer) < self.batch_size:
            return
        # randomly sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*sample))

        next_state_batch = torch.cat([torch.zeros(1, self.num_states, device=self.device)
                                      if ns is None else ns for ns in batch.next_state])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a): compute Q(s) and select taken actions as state action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s_{t+1}): \max_a Q(s_{t+1}, a)
        next_state_values = torch.max(self.target_net(next_state_batch), 1)[0]

        # Expected Q(s,a): r + \gamma V(s_{t+1}) using Bellman equation
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # TD error: Q(s_t, a) - Expected Q(s, a)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                self.net_soft_update()
                duration += 1

            episode_durations.append(duration + 1)
            print("step: {}, rew: {}, duration: {}".format(i + 1, episode_rewards, duration + 1))
            plot_rewards(show_result=False, episode_durations=episode_durations)

        plot_rewards(show_result=True, episode_durations=episode_durations)


if __name__ == "__main__":
    EPSILON_START = 0.9         # start epsilon for greedy policy
    EPSILON_END = 0.05          # minimum epsilon
    EPSILON_DECAY = 1000        # decay rate from epsilon
    BATCH_SIZE = 128            # sample batch size
    GAMMA = 0.99                # reward discount
    TAU = 0.005                 # target network soft update rate
    LR = 1e-4                   # learning rate
    MEMORY_CAPACITY = 3000      # replay buffer memory capacity
    NUM_EPISODES = 500          # number of episodes for sampling and training
    HIDDEN_DIM = 128            # hidden dimension size for Q-network

    LOAD_MODEL_PATH = "saved_models/rl/dqn.pt"
    SAVE_MODEL_PATH = "saved_models/rl/new_dqn.pt"

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
              tau=TAU,
              lr=LR,
              mem_capacity=MEMORY_CAPACITY,
              batch_size=BATCH_SIZE,
              device=run_device,
              )

    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        dqn.load(LOAD_MODEL_PATH)
    dqn.learn(NUM_EPISODES)
    if SAVE_MODEL_PATH:
        dqn.save(SAVE_MODEL_PATH)
