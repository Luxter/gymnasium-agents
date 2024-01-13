# Training script for DQN based on CleanRL implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

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
        logger.debug(f"{single_observation_space_shape}, {single_action_space_shape}")
        self.network = nn.Sequential(
            nn.Linear(np.array(single_observation_space_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, single_action_space_shape),
        )

    def forward(self, x):
        return self.network(x)


def main(total_timesteps: int = 500000):
    seed = 0
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This needs to be a vectorized environment because replay buffer expects batched data
    env = gym.vector.SyncVectorEnv([lambda: gym.make("Acrobot-v1")])

    q_network = QNetwork(env.single_observation_space.shape, env.single_action_space.n)

    buffer_size: Final[int] = 10000

    replay_buffer = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device=device,
        handle_timeout_termination=False,
    )

    # Start
    observations, _ = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        logger.info(f"Global step: {global_step}")

        # Action logic
        q_values = q_network(torch.Tensor(observations))
        actions = torch.argmax(q_values, dim=1).numpy()

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        replay_buffer.add(
            observations, next_observations, actions, rewards, terminations, infos
        )

    if terminations or truncations:
        observations, _ = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
