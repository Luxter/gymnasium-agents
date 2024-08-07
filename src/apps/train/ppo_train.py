# Training script for PPO based on CleanRL implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

import time
from inspect import currentframe
from pathlib import Path
from urllib.parse import urlparse

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from loguru import logger
from torch.distributions import Categorical

from lib.logging import get_function_parameters
from lib.seeding import set_seed


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, single_observation_space_shape: tuple[int], single_action_space_shape: np.ndarray) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(single_observation_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(single_observation_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, single_action_space_shape), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def main(
    exp_name: str = Path(__file__).stem,  # Experiment name
    seed: int = 0,  # Random seed
    env_id: str = "Acrobot-v1",  # Environment ID
    total_timesteps: int = 500000,  # Total number of timesteps
    learning_rate: float = 2.5e-4,  # Learning rate of optimizer
    num_envs: int = 4,  # Number of parallel environments
    num_steps: int = 128,  # Number of steps in each environment per policy rollout
    anneal_lr: bool = True,  # Whether to anneal the learning rate
    gamma: float = 0.99,  # Discount factor gamma
    gae_lambda: float = 0.95,  # Generalized advantage estimation lambda
    num_minibatches: int = 4,  # Number of minibatches to split the batch
    update_epochs: int = 4,  # Number of epochs to update the policy
    norm_adv: bool = True,  # Whether to normalize the advantages
    clip_coef: float = 0.2,  # Surrogate clipping coefficient
    clip_vloss: bool = True,  # Whether to clip the value loss
    ent_coef: float = 0.01,  # Entropy loss coefficient
    vf_coef: float = 0.5,  # Value function loss coefficient
    max_grad_norm: float = 0.5,  # Maximum gradient norm
    target_kl: float | None = None,  # Target KL divergence for early stopping
):
    params = get_function_parameters(currentframe())

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id)) for _ in range(num_envs)]
    )
    envs.action_space.seed(seed)

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for iteration in range(1, num_iterations + 1):
            # Learning rate annealing
            if anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = learning_rate * frac
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_done = np.logical_or(terminations, truncations)
                next_obs = torch.Tensor(next_obs).to(device)
                rewards[step] = torch.Tensor(reward).to(device).view(-1)
                next_done = torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            mlflow.log_metric("episodic_return", info["episode"]["r"], step=global_step)
                            mlflow.log_metric("episodic_length", info["episode"]["l"], step=global_step)

            # Generalized advantage estimation calculation
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
                returns = advantages + values

            # Flatten the batch
            # b means "batch"
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs = []

            for _ in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    # mb means "mini batch"
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Approximate KL divergence and fraction clipped
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -clip_coef, clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()

                # Early stopping
                if target_kl is not None and approx_kl > target_kl:
                    break

            y_pred, y_true = b_values, b_returns
            var_y = torch.var(y_true)
            explained_var = torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

            sps = int(global_step / (time.time() - start_time))

            mlflow.log_metric("charts/learning_rate", optimizer.param_groups[0]["lr"], step=global_step)
            mlflow.log_metric("loss/policy", pg_loss.item(), step=global_step)
            mlflow.log_metric("loss/value", v_loss.item(), step=global_step)
            mlflow.log_metric("loss/entropy", entropy.mean().item(), step=global_step)
            mlflow.log_metric("loss/total", loss.item(), step=global_step)
            mlflow.log_metric("loss/approx_kl_old", old_approx_kl.item(), step=global_step)
            mlflow.log_metric("loss/approx_kl", approx_kl.item(), step=global_step)
            mlflow.log_metric("loss/clipfrac", np.mean(clipfracs), step=global_step)
            mlflow.log_metric("loss/explained_variance", explained_var.item(), step=global_step)
            logger.info(f"SPS: {sps}")
            mlflow.log_metric("charts/SPS", sps, step=global_step)

        artifact_path = urlparse(mlflow.get_artifact_uri()).path
        model_path = Path(f"{artifact_path}/model.pt")
        logger.success(f"Saving model to {model_path}")
        torch.save(agent.state_dict(), model_path)

    envs.close()


if __name__ == "__main__":
    typer.run(main)
