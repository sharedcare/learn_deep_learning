from collections import namedtuple
from typing import Tuple, Optional
from torch import Tensor
from torch.distributions import MultivariateNormal, Categorical

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


class ReplayMemory:
    """Rollout memory for PPO

    """
    Transition = namedtuple('Transition',
                            ('states', 'actions', 'logprobs', 'rewards', 'values', 'dones'))

    def __init__(self, device: torch.device = "cpu") -> None:
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = torch.zeros(1, dtype=torch.float32, device=device)
        self.returns = torch.zeros(1, dtype=torch.float32, device=device)
        self.count = 0
        self.device = device

    def push(self, *transition: Tensor):
        """ save transition to buffer """
        sample = self.Transition(*transition)
        self.states.append(sample.states)
        self.actions.append(sample.actions)
        self.logprobs.append(sample.logprobs)
        self.rewards.append(sample.rewards)
        self.values.append(sample.values)
        self.dones.append(sample.dones)
        self.count += 1

    def sample(self, batch_size: int):
        """ sample transitions from memory buffer """
        batch_start = torch.arange(0, len(self), batch_size)
        indices = torch.randperm(len(self), requires_grad=False, device=self.device)
        batches = [indices[i:i + batch_size] for i in batch_start]
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        logprobs = torch.cat(self.logprobs)
        rewards = torch.cat(self.rewards)
        values = torch.cat(self.values)
        dones = torch.cat(self.dones)

        for batch in batches:
            yield (states[batch], actions[batch], logprobs[batch], rewards[batch],
                   values[batch], dones[batch], self.advantages[batch].detach(), self.returns[batch].detach())

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.count = 0

    def compute_returns(self, last_values: Tensor, gamma: float, lam: float) -> None:
        """ GAE """
        rewards = torch.cat(self.rewards)
        values = torch.cat(self.values)
        dones = torch.cat(self.dones)
        advantage = 0
        self.advantages = torch.zeros_like(rewards, dtype=torch.float32)
        self.returns = torch.zeros_like(rewards, dtype=torch.float32)
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + next_non_terminal * gamma * next_values - values[step]  # td error
            advantage = delta + next_non_terminal * gamma * lam * advantage  # advantage
            self.returns[step] = advantage + values[step]  # td target

        self.advantages = self.returns - values.squeeze(-1)
        # normalize the advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std(dim=-1) + 1e-7)

    def __len__(self) -> int:
        return len(self.states)


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
                 batch_size: int,
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
        self.batch_size = batch_size
        self.device = device
        self.continuous_action = continuous_action
        self.episode_length = episode_length
        self.replay_mem = ReplayMemory(device=device)
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
        next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

        action = torch.tensor(action, device=self.device).unsqueeze(0)

        self.replay_mem.push(obs, action, logprob, reward, value, torch.tensor(done, dtype=torch.int64).unsqueeze(0))
        return next_obs, action, reward, value, done

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs.unsqueeze(0)

    def update(self) -> None:
        for _ in range(self.num_learning_epochs):
            batches = self.replay_mem.sample(self.batch_size)

            for state_batch, action_batch, action_logprobs_batch, reward_batch, value_batch, \
                    done_batch, advantage_batch, return_batch in batches:

                action_logprobs, values, entropy = self.evaluate(state_batch, action_batch)

                # ratio = (pi_theta / pi_theta_old)
                ratios = torch.exp(action_logprobs - action_logprobs_batch.detach())

                # surrogate loss
                surrogate = -ratios * advantage_batch
                surrogate_clipped = -torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantage_batch
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # value loss
                value_loss = self.value_loss_coef * F.mse_loss(values.squeeze(-1), return_batch).mean()

                # entropy loss
                entropy_loss = self.entropy_coef * entropy.mean()

                loss = surrogate_loss + value_loss - entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.replay_mem.clear()

    def learn(self, num_episodes: int) -> None:
        """training process"""
        # rollout
        episode_rewards = []
        best_rewards = self.env.reward_range[0]
        episode_step = 0
        for i in range(num_episodes):
            done = False
            episode_reward = 0
            state = self.reset()
            duration = 0
            while not done:
                # self.env.render("human")
                next_state, action, reward, value, done = self.step(state)
                episode_reward += reward.item()
                duration += 1
                episode_step += 1
                if episode_step % self.episode_length == 0:
                    next_value = self.actor_critic.evaluate(next_state)
                    self.replay_mem.compute_returns(next_value, self.gamma, self.lam)
                    self.update()

                state = next_state

            episode_rewards.append(episode_reward)
            avg_score = np.mean(episode_rewards[-10:])
            print("step: {}, rew: {}, avg score: {}, duration: {}".format(i + 1, episode_reward, avg_score, duration + 1))
            plot_rewards(show_result=False, episode_rewards=episode_rewards)
            if avg_score > best_rewards:
                best_rewards = avg_score
                if SAVE_MODEL_PATH:
                    ppo.save(SAVE_MODEL_PATH)

        plot_rewards(show_result=True, episode_rewards=episode_rewards)

    def play(self, num_episodes: int) -> None:
        """testing process"""
        for i in range(num_episodes):
            done = False
            episode_reward = 0
            duration = 0
            state = self.reset()
            while not done:
                self.env.render()
                action, _ = self.actor_critic.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                episode_reward += reward
                duration += 1
                next_state = torch.tensor(next_state, device=self.device)
                state = next_state

            print("episode: {}, rew: {}, duration: {}".format(i + 1, episode_reward, duration + 1))


if __name__ == "__main__":
    BATCH_SIZE = 4                # sample batch size
    GAMMA = 0.99                    # reward discount
    LAMBDA = 0.95                   # adv rate
    LR = 1e-4                       # learning rate
    NUM_EPISODES = 500             # number of episodes for sampling and training
    EPISODE_LEN = 20                # total episode steps for each rollout episode
    NUM_LEARNING_EPOCHS = 8         # number of epochs for ppo update
    CLIP_PARAM = 0.2                # clip factor for ppo clip
    HIDDEN_DIM = 256                # hidden dimension size for actor-critic network
    VALUE_LOSS_COEF = 0.5           # value loss coefficient
    ENTROPY_LOSS_COEF = 0.01        # entropy loss coefficient

    LOAD_MODEL_PATH = "/home/tchen/Projects/learn_deep_learning/saved_models/rl/ppo.pt"
    SAVE_MODEL_PATH = "/home/tchen/Projects/learn_deep_learning/saved_models/rl/new_ppo.pt"

    gym_env = gym.make("CartPole-v1", render_mode="human")
    n_states = gym_env.observation_space.shape[0]

    if isinstance(gym_env.action_space, gym.spaces.Discrete):
        is_continuous_action = False
        n_actions = gym_env.action_space.n
    else:
        is_continuous_action = True
        n_actions = gym_env.action_space.shape[0]

    run_device = get_device()

    ppo = PPO(gym_env,
              n_states,
              n_actions,
              HIDDEN_DIM,
              BATCH_SIZE,
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

    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        ppo.load(LOAD_MODEL_PATH)
    # ppo.learn(NUM_EPISODES)
    ppo.play(50)
