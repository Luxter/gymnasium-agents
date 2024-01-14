# Training script for DQN based on CleanRL implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

import random
from typing import Final

import fire
import gymnasium as gym
from loguru import logger
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch
import torch.nn as nn

from lib.seeding import set_seed


class QNetwork(nn.Module):
    def __init__(self, single_observation_space_shape, single_action_space_shape):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(np.array(single_observation_space_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, single_action_space_shape),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_eps: float, end_eps: float, duration: int, t: int):
    slope = (end_eps - start_eps) / duration
    return max(start_eps + slope * t, end_eps)


def main(
    seed: int = 0,
    total_timesteps: int = 500000,
    start_eps: float = 1.0,
    end_eps: float = 0.05,
    exploration_fraction: float = 0.5,
    buffer_size: int = 10000,
    batch_size: int = 128,
    gamma: float = 0.99,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This needs to be a vectorized environments because replay buffer expects batched data
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("Acrobot-v1")])

    q_network = QNetwork(
        envs.single_observation_space.shape, envs.single_action_space.n
    ).to(device)

    replay_buffer = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        handle_timeout_termination=False,
    )

    # Start
    obs, _ = envs.reset(seed=seed)
    for global_step in range(total_timesteps):
        logger.info(f"Global step: {global_step}")

        # Action logic
        epsilon = linear_schedule(
            start_eps, end_eps, exploration_fraction * total_timesteps, global_step
        )
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Handle `final_observation` due to auto-reset of vectorized environments
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # Only start training after buffer is filled
        if global_step > buffer_size:
            batch = replay_buffer.sample(batch_size)

            q_values = q_network(batch.observations)
            next_q_values = q_network(batch.next_observations)

            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            # Zero the Q-values corresponding to terminal states
            target_q_values = batch.rewards + (1 - batch.dones) * gamma * max_next_q_values

    envs.close()


if __name__ == "__main__":
    fire.Fire(main)
