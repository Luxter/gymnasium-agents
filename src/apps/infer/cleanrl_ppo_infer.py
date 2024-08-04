import gymnasium as gym
import torch
import typer
from loguru import logger

from apps.train.ppo_train import Agent


def main(
    agent_path: str,
    render_mode: str,  # Rendering mode
    env_id: str = "Acrobot-v1",  # Environment ID
    total_timesteps: int = 500,  # Total number of timesteps
) -> None:
    assert render_mode in ["human", "rgb_array"], "render_mode must be either human or rgb_array"
    env = gym.make(env_id, render_mode=render_mode)
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(
            env, video_folder="videos/PPO", name_prefix=f"{env_id}", episode_trigger=lambda x: True
        )

    observation, _ = env.reset()

    agent = Agent(env.observation_space.shape, env.action_space.n)
    agent.load_state_dict(torch.load(agent_path))
    logger.success(f"Loaded agent from {agent_path}")

    for _ in range(total_timesteps):
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset()

    env.close()

    logger.success("Inference finished")


if __name__ == "__main__":
    typer.run(main)
