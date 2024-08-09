# Gymnasium agents

![Ruff workflow](https://github.com/Luxter/gymnasium-agents/actions/workflows/ruff.yaml/badge.svg)

This repository contains code of basic Reinforcement Learning algorithms implemented for [Gymnasium environments](https://github.com/Farama-Foundation/Gymnasium).

The implementations are inspired by the [CleanRL repository](https://github.com/vwxyzjn/cleanrl).

# Supported algorithms

## DQN





| <video src="https://github.com/user-attachments/assets/cf9dec3c-a4b2-4fcc-b499-1884c2166e11"/> | <video src="https://github.com/user-attachments/assets/52395ec7-64b0-4c44-96e0-00cac1304496"/> | <video src="https://github.com/user-attachments/assets/61867334-dfc9-4164-bb8f-3811b2819d89"/> |
| - | - | - |
| <video src="https://github.com/user-attachments/assets/38b5fc7b-b371-43b8-bdb7-c902721c8c1b"/> | <video src="https://github.com/user-attachments/assets/08fb7c20-b629-427e-b59b-f46f935beeea"/> | <video src="https://github.com/user-attachments/assets/8aea2634-1164-4e95-8fea-948410036d56"/> |

Paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)

## PPO

| <video src="https://github.com/user-attachments/assets/5eb83f8a-592b-4054-969e-6cb0e9d6ef3d"/> | <video src="https://github.com/user-attachments/assets/553639f6-198e-4b4b-8d7b-960405115925"/> | <video src="https://github.com/user-attachments/assets/2c5ecd9e-114e-478e-89f8-907e5f263e84"/>
| - | - | - |
| <video src="https://github.com/user-attachments/assets/1d61bf3e-82bb-4357-b791-0393151d32eb"/> | <video src="https://github.com/user-attachments/assets/d3eddc48-6340-425d-8672-72730fe1c2a7"/> | <video src="https://github.com/user-attachments/assets/ba83c09f-5b35-4e11-9eb7-de32db39987b"/> |



Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)

# Getting started

Prerequisites:
 - Python >= 3.11

To run DQN training:

```bash
git clone https://github.com/Luxter/gymnasium-agents.git
pip install .
python src/apps/train/dqn_train.py
```

To run DQN inference:
```bash
python src/apps/infer/cleanrl_dqn_infer.py <path_to_trained_model>
```

By default `<path_to_trained_model>` should point to experiment inside `mlruns` directory.

The experiments are tracking using [MLFlow library](https://mlflow.org/docs/latest/tracking.html). To start the MLFlow server locally run:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

It is worth noting that the inference supports both models trained by this repository or by original [CleanRL repository](https://github.com/vwxyzjn/cleanrl).
