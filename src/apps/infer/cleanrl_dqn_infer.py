import gymnasium as gym
import numpy as np
import torch
import typer

from apps.train.dqn_train import QNetwork


def main(
    network_path: str,  # Path to the model.pt
    env_id: str = "Acrobot-v1",  # Environment ID
    total_timesteps: int = 1000,  # Total number of timesteps
    seed: int = 0,  # Random seed
) -> None:
    env = gym.make(env_id, render_mode="human")
    observation, _ = env.reset(seed=seed)

    q_network = QNetwork(env.observation_space.shape, env.action_space.n)
    q_network.load_state_dict(torch.load(network_path), strict=True)

    for _ in range(total_timesteps):
        q_values = q_network(torch.Tensor(observation))
        action = torch.argmax(q_values, dim=0).item()
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    typer.run(main)
