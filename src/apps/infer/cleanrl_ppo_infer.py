import gymnasium as gym
from loguru import logger
import torch
import typer

from apps.train.ppo_train import Agent


def main(
    agent_path: str,
    env_id: str = "Acrobot-v1",  # Environment ID
    total_timesteps: int = 1000,  # Total number of timesteps
    seed: int = 0,  # Random seed
) -> None:
    env = gym.make(env_id, render_mode="human")
    observation, _ = env.reset(seed=seed)

    agent = Agent(env.observation_space.shape, env.action_space.n)
    agent.load_state_dict(torch.load(agent_path))
    logger.success(f"Loaded agent from {agent_path}")

    for _ in range(total_timesteps):
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset(seed=seed)

    env.close()

    logger.success("Inference finished")


if __name__ == "__main__":
    typer.run(main)
