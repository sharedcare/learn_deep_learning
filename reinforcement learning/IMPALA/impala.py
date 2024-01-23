import torch
import gymnasium as gym

from actor_critic import ActorCritic


class RolloutBuffer(object):
    def __init__(self,
                 max_size: int,
                 input_shape: int,
                 num_actions: int,
                 device: torch.device = "cpu") -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, input_shape), device=device)
        self.next_state_memory = torch.zeros((self.mem_size, input_shape), device=device)
        self.action_memory = torch.zeros((self.mem_size, num_actions), device=device)
        self.reward_memory = torch.zeros(self.mem_size, device=device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=device)
        self.device = device

    def store(self,
              state: Tensor,
              action: Tensor,
              reward: Tensor,
              next_state: Tensor,
              done: Tensor) -> None:
        """ save transition to buffer """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1


class Actor:
    def __init__(self, learner):
        self.actor_critic = ActorCritic()
        self.learner = learner

    def reset(self) -> Tensor:
        """env reset"""
        obs, info = self.env.reset()
        obs = torch.tensor(obs, device=self.device)
        return obs.unsqueeze(0)

    def step(self, obs: Tensor) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor, bool]:
        """rollout one step"""
        action, logprob, value = self.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs, device=self.device).unsqueeze(0)

    def perform(self):
        self.env = gym.make()
        torch.manual_seed()

        state = self.reset()
        done = False
        rewards = 0
        episode_length = 0

        while True:
            self.actor_critic.load_state_dict(self.learner.actor_critic.state_dict())



class Learner:
    def __init__(self):
        pass


class Impala:
    def __init__(self) -> None:
        self.actor = ActorCritic()
        self.learner = ActorCritic()

    def learn(self):
        pass
