import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
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
    def __init__(self, idx, env_id, learner, device):
        self.idx = idx
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
        while True:
            states, actions, rewards, old_logprobs, old_logits, dones = self.batch_manager.get()
            if len(states) < self.batch_size:
                continue
            logprobs, values, logits = self.actor_critic.evaluate(states, actions)
            actions, old_logits, dones, rewards = actions[1:], old_logits[1:], dones[1:], rewards[1:]
            vs, advantages = self.v_trace(old_logprobs, logprobs, rewards, values, dones)
            # policy loss
            policy_loss = logprobs * advantages.detach()
            policy_loss = policy_loss.sum()
            # baseline loss
            baseline_loss = self.baseline_coef * 0.5 * (vs.detach() - values[:-1]).pow(2)
            baseline_loss = baseline_loss.sum()
            # entropy loss
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = torch.sum(-policy * log_policy, dim=-1)
            entropy_loss = self.entropy_coef * entropy.sum()
            # total loss
            loss = policy_loss + baseline_loss - entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def v_trace(self,
                old_logprobs,
                target_logprobs,
                rewards,
                values,
                dones):
        # log importance sampling weights $log(\pi(a) / \mu(a))$
        log_rhos = target_logprobs - old_logprobs
        rhos = torch.exp(log_rhos)
        rho_clip = min(self.rho_clip_threshold, rhos)
        coef_clip = torch.min(self.coef_clip_threshold, rhos)
        # v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma_{t-s} * \prod_{i=s}^{t-1} c_i * \delta_t
        # \delta_t = \rho_t * (r_t + \gamma V(x_{t+1}) - V(x_t))
        vs = []
        for s in range(len(rewards)):
            v_s = values[s].clone()
            for t in range(s, len(rewards)):
                delta = rho_clip[t] * (rewards[t] + self.gamma * dones[t] * values[t+1] - values[t])
                v_s += torch.prod(self.gamma * dones[s:t] * coef_clip[s:t]) * delta
            vs.append(v_s)
        vs = torch.cat(vs, dim=0)
        # advantage for policy gradient
        # adv_targ = rho_s * (r_s + \gamma * v_{s+1} - V(x_s))
        advantages = rho_clip * (rewards + self.gamma * dones * torch.cat(vs[1:], values[-1]) - values)
        return vs, advantages


class Impala:
    def __init__(self) -> None:
        self.actors = []
        self.learner = Learner()
        for i in range(num_envs):
            actor = Actor(idx=i)
            self.actors.append(actor)
        self.processes = []

    def train(self):
        batch = mp.Queue(maxsize=1)
        for idx, actor in enumerate(self.actors):
            p = mp.Process(target=actor.perform, args=(idx, batch))
            p.start()
            self.processes.append(p)

        learner = mp.Process(target=self.learner.learn, args=(batch,))
        learner.start()
        self.processes.append(learner)

        for p in self.processes:
            p.join()

