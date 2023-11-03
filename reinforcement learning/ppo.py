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

    def __init__(self, device: str = "cpu") -> None:
        self.mem = []
        self.count = 0

    def push(self, *transition: Tensor):
        """ save transition to buffer """
        self.mem.append(self.Transition(*transition))
        self.count += 1

    def sample(self, batch_size: int = None) -> Transition:
        """ sample transitions from memory buffer """
        if batch_size is None:
            return self.Transition(*zip(*self.mem))
        else:
            return self.Transition(*zip(*random.sample(self.mem, batch_size)))

    def clear(self) -> None:
        self.mem = []

    def __len__(self) -> int:
        return len(self.mem)


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

        self.distribution = None
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.to(device)

    def forward(self):
        raise NotImplementedError

    @property
    def action_std(self) -> Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        return self.distribution.entropy().sum(dim=-1)

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

    def evaluate(self, obs: Tensor) -> Tensor:
        value = self.critic(obs)
        return value


class PPO:
    def __init__(self,
                 env: gym.Env,
                 num_states: int,
                 num_actions: int,
                 hidden_dim: int,
                 learning_rate: float,
                 episode_length: int,
                 continuous_action: bool,
                 clip_param: float,
                 value_clip_param: float,
                 num_learning_epochs: int,
                 value_loss_coef: float,
                 entropy_coef: float,
                 gamma: float,
                 lam: float,
                 max_grad_norm: float,
                 device: torch.device = "cpu") -> None:
        self.env = env
        self.device = device
        self.continuous_action = continuous_action
        self.episode_length = episode_length
        self.replay_buffer = ReplayBuffer()
        self.actor_critic = ActorCritic(num_states, num_actions, hidden_dim,
                                        continuous_action=continuous_action, device=device)
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

    def save(self, path: str) -> None:
        """save model to designated path"""
        torch.save(self.actor_critic.state_dict(), path)
        print("model saved at: {}".format(path))

    def load(self, path: str) -> None:
        """load network from path"""
        self.actor_critic.load_state_dict(torch.load(path))
        print("model loaded from: {}".format(path))

    def act(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action, action_logprobs = self.actor_critic.act(state)
        value = self.actor_critic.evaluate(state)
        # self.replay_buffer.push(state, action, action_logprobs)
        if self.continuous_action:
            return action.detach(), action_logprobs, value
        else:
            return action.item(), action_logprobs, value

    def evaluate(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _, _ = self.actor_critic.act(state)
        value = self.actor_critic.evaluate(state)
        action_logprob = self.actor_critic.distribution.log_prob(action)
        entropy = self.actor_critic.entropy

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

        action = torch.tensor(action, device=self.device).unsqueeze(0)

        self.replay_buffer.push(obs, action, logprob, reward, value, next_obs)
        return next_obs, action, reward, value, done

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs.unsqueeze(0)

    def compute_returns(self, last_state: Tensor, values: Tensor,
                        rewards: Tensor, dones: Tensor) -> Tuple[Tensor, Tensor]:
        """ GAE """
        last_values = self.actor_critic.evaluate(last_state)
        advantage = 0
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + next_non_terminal * self.gamma * next_values - values[step]  # td error
            advantage = delta + next_non_terminal * self.gamma * self.lam * advantage  # advantage
            returns[step] = advantage + values[step]  # td target
            advantages[step] = advantage
            # normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std(dim=-1) + 1e-7)

        return advantages, returns

    def update(self) -> None:
        batch = self.replay_buffer.sample()

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        action_logprobs_batch = torch.cat(batch.logprobs)
        reward_batch = torch.cat(batch.reward)
        value_batch = torch.cat(batch.value)
        # next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor([0. if ns is None else 1. for ns in batch.next_state], device=self.device)

        advantages, returns = self.compute_returns(state_batch[-1], value_batch, reward_batch, done_batch)

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

            # loss = surrogate_loss + value_loss - entropy_loss

            self.optimizer.zero_grad()
            surrogate_loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()

        self.replay_buffer.clear()

    def learn(self, num_episodes: int) -> None:
        """training process"""
        # rollout
        episode_durations = []
        for i in range(num_episodes):
            step = 0
            while step < self.episode_length:
                done = False
                episode_rewards = 0
                state = self.reset()
                duration = 0
                while not done:
                    next_state, action, reward, value, done = self.step(state)
                    episode_rewards += reward.item()
                    state = next_state
                    duration += 1
                    step += 1

                episode_durations.append(duration + 1)
                print("step: {}, rew: {}, duration: {}".format(i + 1, episode_rewards, duration + 1))
            self.update()
            plot_rewards(show_result=False, episode_durations=episode_durations)

        plot_rewards(show_result=True, episode_durations=episode_durations)


if __name__ == "__main__":
    BATCH_SIZE = 128                # sample batch size
    GAMMA = 0.99                    # reward discount
    LAMBDA = 0.005                  # adv rate
    LR = 1e-4                       # learning rate
    NUM_EPISODES = 500              # number of episodes for sampling and training
    EPISODE_LEN = 40                # total episode steps for each rollout episode
    NUM_LEARNING_EPOCHS = 8         # number of epochs for ppo update
    CLIP_PARAM = 0.2                # clip factor for ppo clip
    HIDDEN_DIM = 128                # hidden dimension size for actor-critic network
    VALUE_LOSS_COEF = 0.5           # value loss coefficient
    ENTROPY_LOSS_COEF = 0.01        # entropy loss coefficient

    LOAD_MODEL_PATH = "saved_models/rl/ppo.pt"
    SAVE_MODEL_PATH = "saved_models/rl/new_ppo.pt"

    gym_env = gym.make("CartPole-v1").unwrapped
    n_states = gym_env.observation_space.shape[0]
    n_actions = gym_env.action_space.n

    if isinstance(gym_env.action_space, gym.spaces.Discrete):
        is_continuous_action = False
    else:
        is_continuous_action = True

    run_device = get_device()

    ppo = PPO(gym_env,
              n_states,
              n_actions,
              HIDDEN_DIM,
              LR,
              EPISODE_LEN,
              is_continuous_action,
              CLIP_PARAM,
              CLIP_PARAM,
              NUM_LEARNING_EPOCHS,
              VALUE_LOSS_COEF,
              ENTROPY_LOSS_COEF,
              GAMMA,
              LAMBDA,
              max_grad_norm=1.0,
              device=run_device,
              )

    ppo.learn(NUM_EPISODES)
