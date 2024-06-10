import os.path
import sys
import random
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import gym
from mlx.core import array
from collections import deque, namedtuple
from typing import List, Tuple, Optional

sys.path.append("./")

from utils import plot_rewards


class ReplayBuffer(object):
    Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

    def __init__(self, capacity: int) -> None:
        self.mem = deque([], maxlen=capacity)

    def push(self, *transition: array) -> None:
        """save transition to buffer"""
        self.mem.append(self.Transition(*transition))

    def sample(self, batch_size: int) -> list:
        """randomly sample memory buffer"""
        return random.sample(self.mem, batch_size)

    def __len__(self) -> int:
        return len(self.mem)


class Net(nn.Module):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_dim: int,
        device: mx.Device = mx.default_device,
    ) -> None:
        super(Net, self).__init__()
        layer_sizes = [num_states] + [hidden_dim] * 2 + [num_actions]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.device = device

    def __call__(self, x: array) -> array:
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)

    def load_weights_dict(self, model: nn.Module) -> None:
        for i, layer in enumerate(model.layers):
            weight = [
                ("weight", layer.parameters()["weight"]),
                ("bias", layer.parameters()["bias"]),
            ]
            self.layers[i].load_weights(weight)

    def loss_fn(self, state: array, action: array, expected_state_action_values: array) -> array:
        # Q(s_t, a): compute Q(s) and select taken actions as state action values
        state_action_values = mx.take_along_axis(self(state), action, 1)
        losses = nn.losses.smooth_l1_loss(
            state_action_values.squeeze(1), expected_state_action_values
        )
        return losses


class DQN:
    def __init__(
        self,
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
        device: mx.Device = mx.default_device,
    ) -> None:
        super(DQN, self).__init__()
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.policy_net = Net(num_states, num_actions, hidden_dim, device)
        self.target_net = Net(num_states, num_actions, hidden_dim, device)
        self.target_net.load_weights_dict(self.policy_net)
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
        self.optimizer = optim.Adam(learning_rate=self.lr)

        self.steps_done = 0

        mx.eval(self.policy_net.parameters())

    def save(self, path: str) -> None:
        """save policy model to designated path"""
        self.policy_net.save_weights(path)
        print("Policy model saved at: {}".format(path))

    def load(self, path: str) -> None:
        """load network from path"""
        self.policy_net.load_weights(path)
        self.target_net.load_weights(self.policy_net.parameters())
        print("Policy model loaded from: {}".format(path))

    def act(self, obs: array) -> array:
        """choose action"""
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > epsilon:
            action_value = self.policy_net(obs)
            action = mx.argmax(action_value, 1)
            return mx.expand_dims(action, 0)
        else:
            return mx.array([[self.env.action_space.sample()]])

    def step(self, obs: array) -> Tuple[Optional[array], array, array, bool]:
        """rollout one step"""
        action = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action.item())
        reward = mx.array([reward])
        done = terminated or truncated
        if done:
            next_obs = None
        else:
            next_obs = mx.expand_dims(array(next_obs), 0)

        self.replay_buffer.push(obs, action, reward, next_obs)
        return next_obs, action, reward, done

    def reset(self) -> array:
        """env reset"""
        obs, info = self.env.reset()
        obs = mx.array(obs)
        return mx.expand_dims(obs, 0)

    def net_soft_update(self):
        """
        soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′

        """
        for i in range(len(self.target_net.parameters().copy()["layers"])):
            update_weight = (
                "weight",
                self.target_net.layers[i].parameters()["weight"] * (1 - self.tau)
                + self.policy_net.layers[i].parameters()["weight"] * self.tau,
            )
            update_bias = (
                "bias",
                self.target_net.layers[i].parameters()["bias"] * (1 - self.tau)
                + self.policy_net.layers[i].parameters()["bias"] * self.tau,
            )
            self.target_net.layers[i].load_weights([update_weight, update_bias])

    def update(self) -> array:
        """update policy network"""
        if len(self.replay_buffer) < self.batch_size:
            return
        # randomly sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*sample))

        next_state_batch = mx.concatenate(
            [
                mx.zeros((1, self.num_states)) if ns is None else ns
                for ns in batch.next_state
            ]
        )
        state_batch = mx.concatenate(batch.state)
        action_batch = mx.concatenate(batch.action)
        reward_batch = mx.concatenate(batch.reward)

        # V(s_{t+1}): \max_a Q(s_{t+1}, a)
        next_state_values = mx.max(self.target_net(next_state_batch), 1)

        # Expected Q(s,a): r + \gamma V(s_{t+1}) using Bellman equation
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # TD error: Q(s_t, a) - Expected Q(s, a)
        loss_fn = self.policy_net.loss_fn
        loss_and_grad_fn = nn.value_and_grad(self.policy_net, loss_fn)
        loss, grads = loss_and_grad_fn(
            state_batch, action_batch, expected_state_action_values
        )

        self.optimizer.update(self.policy_net, grads)
        del grads
        mx.eval(self.policy_net.parameters())
        return loss

    def learn(self, num_episodes: int) -> None:
        """training process"""
        episode_rewards = []
        best_rewards = self.env.reward_range[0]
        for i in range(num_episodes):
            done = False
            episode_reward = 0
            state = self.reset()
            duration = 0
            while not done:
                next_state, action, reward, done = self.step(state)
                episode_reward += reward.item()
                state = next_state
                loss = self.update()
                self.net_soft_update()
                duration += 1

            episode_rewards.append(episode_reward)
            avg_score = np.mean(episode_rewards[-100:])
            print(
                "step: {}, rew: {:0f}, avg score: {:2f}, duration: {}, loss: {:5f}".format(
                    i + 1, episode_reward, avg_score, duration + 1, loss.item() if loss else 0
                )
            )
            plot_rewards(show_result=False, episode_rewards=episode_rewards)
            if avg_score > best_rewards:
                best_rewards = avg_score
                if SAVE_MODEL_PATH:
                    dqn.save(SAVE_MODEL_PATH)

        plot_rewards(show_result=True, episode_rewards=episode_rewards)


if __name__ == "__main__":
    EPSILON_START = 0.9  # start epsilon for greedy policy
    EPSILON_END = 0.05  # minimum epsilon
    EPSILON_DECAY = 1000  # decay rate from epsilon
    BATCH_SIZE = 128  # sample batch size
    GAMMA = 0.99  # reward discount
    TAU = 0.005  # target network soft update rate
    LR = 1e-4  # learning rate
    MEMORY_CAPACITY = 3000  # replay buffer memory capacity
    NUM_EPISODES = 500  # number of episodes for sampling and training
    HIDDEN_DIM = 128  # hidden dimension size for Q-network

    LOAD_MODEL_PATH = "saved_models/rl/dqn.npz"
    SAVE_MODEL_PATH = "saved_models/rl/new_dqn.npz"

    gym_env = gym.make("CartPole-v1").unwrapped
    n_states = gym_env.observation_space.shape[0]
    n_actions = gym_env.action_space.n

    run_device = mx.default_device

    dqn = DQN(
        env=gym_env,
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

    # if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
    #     dqn.load(LOAD_MODEL_PATH)
    dqn.learn(NUM_EPISODES)
