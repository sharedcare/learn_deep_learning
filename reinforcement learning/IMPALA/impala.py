import torch
import torch.optim as optim
import gymnasium as gym

from torch import Tensor
from typing import Tuple
from actor_critic import ActorCritic


class RolloutBuffer(object):
    def __init__(
        self,
        max_size: int,
        input_shape: int,
        num_actions: int,
        device: torch.device = "cpu",
    ) -> None:
        self.mem_size = max_size
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, input_shape), device=device)
        self.next_state_memory = torch.zeros(
            (self.mem_size, input_shape), device=device
        )
        self.action_memory = torch.zeros((self.mem_size, num_actions), device=device)
        self.reward_memory = torch.zeros(self.mem_size, device=device)
        self.terminal_memory = torch.zeros(
            self.mem_size, dtype=torch.bool, device=device
        )
        self.device = device

    def reset(self):
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, self.input_shape), device=self.device)
        self.next_state_memory = torch.zeros(
            (self.mem_size, self.input_shape), device=self.device
        )
        self.action_memory = torch.zeros((self.mem_size, self.num_actions), device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, device=self.device)
        self.terminal_memory = torch.zeros(
            self.mem_size, dtype=torch.bool, device=self.device
        )

    def store(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """save transition to buffer"""
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def __len__(self):
        return self.mem_cntr


class Actor:
    def __init__(self, env_id, learner, device):
        self.env = gym.make(env_id)
        self.rollout_buffer = RolloutBuffer()
        self.actor_critic = ActorCritic()
        self.learner = learner
        self.device = device

    def act(self, state: Tensor) -> Tuple[Tensor, ...]:
        action, action_logprobs = self.actor_critic.act(state)
        logits = self.actor_critic.distribution.logits
        return action, action_logprobs, logits

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs.unsqueeze(0)

    def step(
        self, obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        """rollout one step"""
        action, logprobs, logits = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

        self.rollout_buffer.store(obs, action, reward, next_obs, Tensor(done))
        return next_obs, action, logits, reward, done

    def perform(self):
        torch.manual_seed(self.seed)

        state = self.reset()
        rewards = 0
        done = False
        num_episodes = 0
        total_episode_length = 0

        iterations = 0
        timesteps = 0

        while True:
            self.actor_critic.load_state_dict(self.learner.actor_critic.state_dict())
            self.rollout_buffer.reset()
            with torch.no_grad():
                for step in range(self.num_steps):
                    next_state, action, logits, reward, done = self.step(state)
                    rewards += reward
                    state = next_state
                    timesteps += 1
                    if done:
                        num_episodes += 1
                        total_episode_length += 1

            if done:
                state = self.reset()
                if timesteps >= self.total_num_steps:
                    iterations += 1
                    rewards = 0
                    num_episodes = 0
                    timesteps = 0


class Learner:
    def __init__(self, lr, batch_manager):
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.actor_critic.share_memory()
        self.batch_manager = batch_manager

    def learn(self):
        torch.manual_seed(self.seed)
        batch = []
        best = 0

        while True:
            states, actions, rewards, old_logprobs, old_logits, dones = self.batch_manager.get()
            # batch.append(trajectory)
            policy_loss = 0.
            value_loss = 0.
            entropy_loss = 0.
            if len(states) < self.batch_size:
                continue
            logprobs, values, logits = self.actor_critic.evaluate(states, actions)
            actions, old_logits, dones, rewards = actions[1:], old_logits[1:], dones[1:], rewards[1:]
            logits, values = logits[:-1], values[:-1]

    def v_trace(self, policy_logits, target_logits, actions, rewards, values):
        pass


class Impala:
    def __init__(self) -> None:
        self.actor = ActorCritic()
        self.learner = ActorCritic()

    def learn(self):
        pass
