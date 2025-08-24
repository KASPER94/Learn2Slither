# This file is used to create a configuration for the DQN agent.
# It is used to create a configuration for the DQN agent.
# It is used to create a configuration for the DQN agent.
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from learn2slither.DQNAgent.model import Linear_QNet

@dataclass
class Agent_config:
    n_games: int
    input_layer_size: int
    hidden_layer_size: int
    output_layer_size: int
    lr: float
    gamma: float
    memory: deque
    batch_size: int
    train_iters_per_game: int
    epsilon_start: float
    epsilon: float
    epsilon_decay: float
    epsilon_end: float
    model: Linear_QNet
    target_model: Linear_QNet
    optimizer: optim.Optimizer
    criterion: nn.Module

def make_agent(trial):
    default_config = Agent_config(
        n_games=0,
        input_layer_size=11,
        hidden_layer_size=256,
        output_layer_size=3,
        lr=0.001,
        gamma=0.9,
        memory=deque(maxlen=100000),
        batch_size=1000,
        train_iters_per_game=4,
        epsilon_start=0.9,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_end=0.01,
        model=Linear_QNet(input_layer_size, hidden_layer_size, output_layer_size),
        target_model=Linear_QNet(input_layer_size, hidden_layer_size, output_layer_size),
        optimizer=optim.Adam(model.parameters(), lr=lr),
        criterion=nn.MSELoss(),
    )

    config = {
        'n_games': trial.suggest_int('n_games', 0, 1000),
        'input_layer_size': trial.suggest_int('input_layer_size', 1, 100),
        'hidden_layer_size': trial.suggest_int('hidden_layer_size', 1, 100),
        'output_layer_size': trial.suggest_int('output_layer_size', 1, 100),
        'lr': trial.suggest_float('lr', 0.0001, 0.01, step=0.0001),
        'gamma': trial.suggest_float('gamma', 0.8, 0.99, step=0.01),
        'batch_size': trial.suggest_int('batch_size', 100, 1000, step=100),
        'train_iters_per_game': trial.suggest_int('train_iters_per_game', 1, 10),
        'epsilon_start': trial.suggest_float('epsilon_start', 0.8, 0.99, step=0.01),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.8, 0.99, step=0.01),
        'epsilon_end': trial.suggest_float('epsilon_end', 0.01, 0.1, step=0.01),
    }
    config = {**default_config, **config}
    return Agent_config(**config)

