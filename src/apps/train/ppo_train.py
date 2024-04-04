# Training script for PPO based on CleanRL implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

from pathlib import Path
import time

import gymnasium as gym
from loguru import logger
import mlflow
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim
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
    total_timesteps: int = 5000,  # Total number of timesteps
    learning_rate: float = 2.5e-4,  # Learning rate of optimizer
    num_envs: int = 4,  # Number of parallel environments
    num_steps: int = 128,  # Number of steps in each environment per policy rollout
):
    batch_size = int(num_envs * num_steps)
    num_iterations = total_timesteps // batch_size
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Does this need to be vectorized environment?
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id)) for _ in range(num_envs)]
    )
    envs.action_space.seed(seed)

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start
    global_step = 0
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        for iteration in range(1, num_iterations + 1):
            for step in range(num_steps):
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_done = np.logical_or(terminations, truncations)
                next_obs = torch.Tensor(next_obs).to(device)
                rewards[step] = torch.Tensor(reward).to(device)
                next_done = torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    info = infos["final_info"][0]
                    logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    mlflow.log_metric("episodic_return", info["episode"]["r"], step=global_step)
                    mlflow.log_metric("episodic_length", info["episode"]["l"], step=global_step)


if __name__ == "__main__":
    typer.run(main)