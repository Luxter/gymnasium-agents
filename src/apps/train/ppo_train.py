# Training script for PPO based on CleanRL implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import typer

from lib.seeding import set_seed

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, single_observation_space_shape: tuple[int], single_action_space_shape: np.ndarray) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(single_observation_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(single_observation_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, single_action_space_shape), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def main(
    exp_name: str = Path(__file__).stem,  # Experiment name
    seed: int = 0,  # Random seed
    env_id: str = "Acrobot-v1",  # Environment ID
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Does this need to be vectorized environment?
    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))])
    envs.action_space.seed(seed)


if __name__ == "__main__":
    typer.run(main)
