import gymnasium as gym
from loguru import logger
import numpy as np
import torch
import typer

from apps.train.dqn_train import QNetwork


def main(
    network_path: str,  # Path to the model.pt
    render_mode: str,  # Rendering mode
    env_id: str = "Acrobot-v1",  # Environment ID
    total_timesteps: int = 500,  # Total number of timesteps
) -> None:
    assert render_mode in ["human", "rgb_array"], "render_mode must be either human or rgb_array"
    env = gym.make(env_id, render_mode=render_mode)
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(
            env, video_folder="videos/DQN", name_prefix=f"{env_id}", episode_trigger=lambda x: True
        )

    observation, _ = env.reset()

    q_network = QNetwork(env.observation_space.shape, env.action_space.n)
    q_network.load_state_dict(torch.load(network_path), strict=True)
    logger.success(f"Loaded agent from {network_path}")

    for _ in range(total_timesteps):
        q_values = q_network(torch.Tensor(observation))
        action = torch.argmax(q_values, dim=0).item()
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset()

    env.close()

    logger.success("Inference finished")


if __name__ == "__main__":
    typer.run(main)
