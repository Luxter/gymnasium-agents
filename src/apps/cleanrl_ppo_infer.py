import fire
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, single_observation_space_shape, single_action_space_shape):
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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def main(agent_path: str):
    seed = 0

    env = gym.make("Acrobot-v1", render_mode="human")
    observation, _ = env.reset(seed=seed)

    agent = Agent(env.observation_space.shape, env.action_space.n)
    agent.load_state_dict(torch.load(agent_path))

    steps_count = 1000
    for _ in range(steps_count):
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
