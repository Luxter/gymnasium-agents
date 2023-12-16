import fire
import gymnasium as gym
import numpy as np
import torch

from apps.train.dqn_train import QNetwork


def main(network_path: str, steps_count: int = 1000):
    seed = 0

    env = gym.make("Acrobot-v1", render_mode="human")
    observation, _ = env.reset(seed=seed)

    q_network = QNetwork(env.observation_space.shape, env.action_space.n)
    q_network.load_state_dict(torch.load(network_path), strict=True)

    for _ in range(steps_count):
        q_values = q_network(torch.Tensor(observation))
        action = torch.argmax(q_values, dim=0).item()
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
