"""Trainer pour DQN Agent avec Snake Environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from .env import SnakeEnv
from .DQNAgent import DQNAgent
from .rewards import compute_reward
from .state import encode_state


@dataclass
class DQNTrainConfig:
    episodes: int = 100
    max_steps: int = 500
    headless: bool = True
    verbose: bool = False
    save_model_path: Optional[str] = None
    load_model_path: Optional[str] = None
    save_every: int = 0  # 0 = sauvegarde uniquement en fin si chemin fourni


class DQNTrainer:
    def __init__(self, env: Optional[SnakeEnv] = None, agent: Optional[DQNAgent] = None) -> None:
        self.env = env or SnakeEnv(10)  # data from main.py par défaut
        self.agent = agent or DQNAgent(state_size=128, action_size=3)  # data from state encoder

    def _maybe_load(self, cfg: DQNTrainConfig) -> None:
        if cfg.load_model_path:
            self.agent.load(cfg.load_model_path)

    def _maybe_save(self, cfg: DQNTrainConfig) -> None:
        if cfg.save_model_path:
            self.agent.save(cfg.save_model_path)

    def train(self, cfg: DQNTrainConfig) -> None:
        """Boucle d'entraînement DQN avec replay memory."""
        self._maybe_load(cfg)
        max_length_seen = 0  # data from user request: track max length during training

        for ep in range(1, cfg.episodes + 1):
            self.env.reset()  # data from SnakeEnv.reset()
            s = self._state_to_vector(encode_state(self.env.observation()))
            episode_reward = 0.0
            steps = 0

            for _ in range(cfg.max_steps):
                # data from DQN: choose action and get next state
                a = self.agent.choose_action(s)
                step = self.env.step(a)
                s2 = self._state_to_vector(encode_state(self.env.observation()))

                r = compute_reward(
                    died=step.done,
                    ate_green=step.grew,
                    ate_red=step.shrank,
                )
                episode_reward += r
                
                # data from DQN: store experience in replay memory
                self.agent.remember(s, a, r, s2, step.done)
                
                # data from DQN: train on batch
                self.agent.replay()
                
                s = s2
                steps += 1
                if step.done:
                    break

            self.agent.end_episode()

            # data from user request: update max length seen
            current_length = len(self.env.snake)
            if current_length > max_length_seen:
                max_length_seen = current_length

            if cfg.verbose:
                # data from env: score = len(snake) - 3
                score = self.env.score
                length = len(self.env.snake)
                print(
                    f"Episode {ep:4d} | steps={steps:3d} | score={score:3d} | "
                    f"length={length:2d} | max_length={max_length_seen:2d} | return={episode_reward:7.2f} | eps={self.agent.epsilon:.3f}"
                )

            if cfg.save_model_path and cfg.save_every > 0 and ep % cfg.save_every == 0:
                self._maybe_save(cfg)

        # Sauvegarde finale si un chemin est fourni et pas de périodicité
        if cfg.save_model_path and (cfg.save_every == 0 or cfg.episodes % cfg.save_every != 0):
            self._maybe_save(cfg)

    def _state_to_vector(self, state_int: int) -> np.ndarray:
        """Convertit l'état entier en vecteur one-hot de 128 dimensions."""
        # data from state encoder: state_int is 0-127
        vector = np.zeros(128, dtype=np.float32)
        vector[state_int] = 1.0
        return vector
    
    def evaluate(self, episodes: int = 100, max_steps: int = 100) -> None:
        """Boucle d'évaluation sans apprentissage."""
        self.agent.model.eval()  # data from PyTorch: set to evaluation mode
        try:
            for _ in range(episodes):
                self.env.reset()
                s = self._state_to_vector(encode_state(self.env.observation()))
                for _ in range(max_steps):
                    a = self.agent.choose_action(s)
                    step = self.env.step(a)
                    s = self._state_to_vector(encode_state(self.env.observation()))
                    if step.done:
                        break
        finally:
            self.agent.model.train()  # data from PyTorch: set back to training mode 