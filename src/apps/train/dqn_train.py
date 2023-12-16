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

    env = gym.make("Acrobot-v1")

    q_network = QNetwork(env.observation_space.shape, env.action_space.n)

    buffer_size: Final[int] = 10000

    replay_buffer = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device=device,
        handle_timeout_termination=False,
    )

    print(total_timesteps, type(total_timesteps))

    # Start
    observation, _ = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        logger.info(f"Global step: {global_step}")
        # Action logic
        q_values = q_network(torch.Tensor(observation))
        action = torch.argmax(q_values, dim=0).numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        # This expects ndarrays for all fields.
        # Do I need to use SyncVectorEnv for environment?
        replay_buffer.add(observation, action, reward, terminated, truncated, info)

    if terminated or truncated:
        observation, _ = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
