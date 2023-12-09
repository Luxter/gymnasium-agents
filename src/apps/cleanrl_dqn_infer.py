import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


# ALGO LOGIC: initialize agent here:
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


def main(network_path: str):
    seed = 0

    env = gym.make("Acrobot-v1", render_mode="human")
    observation, _ = env.reset(seed=seed)

    q_network = QNetwork(env.observation_space.shape, env.action_space.n)
    q_network.load_state_dict(torch.load(network_path), strict=True)

    steps_count = 1000
    for _ in range(steps_count):
        q_values = q_network(torch.Tensor(observation))
        action = torch.argmax(q_values, dim=0).item()
        observation, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
