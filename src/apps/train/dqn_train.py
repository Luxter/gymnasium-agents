# Training script for DQN based on CleanRL implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

from pathlib import Path
import random
import time
from urllib.parse import urlparse

import gymnasium as gym
from loguru import logger
import mlflow
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

from lib.seeding import set_seed


class QNetwork(nn.Module):
    def __init__(self, single_observation_space_shape: tuple[int], single_action_space_shape: np.ndarray) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(np.array(single_observation_space_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, single_action_space_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def linear_schedule(start_eps: float, end_eps: float, duration: int, t: int) -> float:
    slope = (end_eps - start_eps) / duration
    return max(start_eps + slope * t, end_eps)


def main(
    exp_name: str = Path(__file__).stem,  # Experiment name
    seed: int = 0,  # Random seed
    env_id: str = "Acrobot-v1",  # Environment ID
    total_timesteps: int = 500000,  # Total number of timesteps
    start_eps: float = 1.0,  # Initial epsilon for exploration
    end_eps: float = 0.05,  # Final epsilon for exploration
    exploration_fraction: float = 0.5,  # Fraction of timesteps to explore
    buffer_size: int = 10000,  # Replay memory buffer size
    learning_rate: float = 2.5e-4,  # Learning rate of optimizer
    tau: float = 1.0,  # Target network update rate
    target_network_frequency: int = 500,  # How often to update target network
    batch_size: int = 128,  # Batch size for training
    gamma: float = 0.99,  # Discount factor,
    train_frequency: int = 10,  # How many episodes accumulated between training steps
) -> None:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This needs to be a vectorized environments because replay buffer expects batched data
    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))])
    envs.action_space.seed(seed)

    q_network = QNetwork(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = QNetwork(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    replay_buffer = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # Start
    obs, _ = envs.reset(seed=seed)

    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        for global_step in range(total_timesteps):
            # Action logic
            epsilon = linear_schedule(start_eps, end_eps, exploration_fraction * total_timesteps, global_step)
            if random.random() < epsilon:
                actions = envs.action_space.sample()
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if "final_info" in infos:
                info = infos["final_info"][0]
                logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                mlflow.log_metric("episodic_return", info["episode"]["r"], step=global_step)
                mlflow.log_metric("episodic_length", info["episode"]["l"], step=global_step)

            # Handle `final_observation` due to auto-reset of vectorized environments
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

            obs = next_obs

            # Only start training after buffer is filled
            if global_step > buffer_size:
                if global_step % train_frequency == 0:
                    batch = replay_buffer.sample(batch_size)

                    q_values = q_network(batch.observations).gather(1, batch.actions).squeeze()

                    with torch.no_grad():
                        target_max, _ = target_network(batch.next_observations).max(dim=1)
                        # Zero the Q-values corresponding to terminal states
                        target_q_values = batch.rewards.flatten() + (1 - batch.dones.flatten()) * gamma * target_max

                    loss = F.mse_loss(q_values, target_q_values)

                    if global_step % 100 == 0:
                        mlflow.log_metric("td_loss", loss, step=global_step)
                        mlflow.log_metric("q_values", q_values.mean().item(), step=global_step)
                        mlflow.log_metric(
                            "SPS",
                            int(global_step / (time.time() - start_time)),
                            step=global_step,
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Soft-update target network
                if global_step % target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        target_network.parameters(), q_network.parameters()
                    ):
                        target_network_param.data.copy_(
                            tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                        )

        artifact_path = urlparse(mlflow.get_artifact_uri()).path
        model_path = Path(f"{artifact_path}/model.pt")
        logger.success(f"Saving model to {model_path}")
        torch.save(q_network.state_dict(), model_path)

    envs.close()


if __name__ == "__main__":
    typer.run(main)
