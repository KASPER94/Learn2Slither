"""Boucle d'entraînement/évaluation (implémentation minimale).

Relie l'environnement, l'encodeur d'état et l'agent Q-learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .env import SnakeEnv
from .q_agent import QAgent
from .rewards import compute_reward
from .state import encode_state


@dataclass
class TrainConfig:
    episodes: int = 100
    max_steps: int = 500
    headless: bool = True
    verbose: bool = False
    save_model_path: Optional[str] = None
    load_model_path: Optional[str] = None
    save_every: int = 0  # 0 = sauvegarde uniquement en fin si chemin fourni


class Trainer:
    def __init__(self, env: Optional[SnakeEnv] = None, agent: Optional[QAgent] = None) -> None:
        self.env = env or SnakeEnv(10)  # data from main.py par défaut
        self.agent = agent or QAgent()

    def _maybe_load(self, cfg: TrainConfig) -> None:
        if cfg.load_model_path:
            self.agent = QAgent.load(cfg.load_model_path)

    def _maybe_save(self, cfg: TrainConfig) -> None:
        if cfg.save_model_path:
            self.agent.save(cfg.save_model_path)

    def train(self, cfg: TrainConfig) -> None:
        """Boucle d'entraînement avec logs optionnels et sauvegardes."""
        self._maybe_load(cfg)

        for ep in range(1, cfg.episodes + 1):
            self.env.reset()  # data from SnakeEnv.reset()
            s = encode_state(self.env.observation())
            episode_reward = 0.0
            steps = 0

            for _ in range(cfg.max_steps):
                a = self.agent.choose_action(s)
                step = self.env.step(a)
                s2 = encode_state(self.env.observation())

                r = compute_reward(
                    died=step.done,
                    ate_green=step.grew,
                    ate_red=step.shrank,
                )
                episode_reward += r
                self.agent.update(s, a, r, s2)
                s = s2
                steps += 1
                if step.done:
                    break

            self.agent.end_episode()

            if cfg.verbose:
                # data from env: score = len(snake) - 3
                score = self.env.score
                length = len(self.env.snake)
                print(
                    f"Episode {ep:4d} | steps={steps:3d} | score={score:3d} | "
                    f"length={length:2d} | return={episode_reward:7.2f} | eps={self.agent.epsilon:.3f}"
                )

            if cfg.save_model_path and cfg.save_every > 0 and ep % cfg.save_every == 0:
                self._maybe_save(cfg)

        # Sauvegarde finale si un chemin est fourni et pas de périodicité
        if cfg.save_model_path and (cfg.save_every == 0 or cfg.episodes % cfg.save_every != 0):
            self._maybe_save(cfg)

    def evaluate(self, episodes: int = 10, max_steps: int = 500) -> None:
        """Boucle d'évaluation sans apprentissage."""
        learning_state = self.agent.learning_enabled
        self.agent.learning_enabled = False
        try:
            for _ in range(episodes):
                self.env.reset()
                s = encode_state(self.env.observation())
                for _ in range(max_steps):
                    a = self.agent.choose_action(s)
                    step = self.env.step(a)
                    s = encode_state(self.env.observation())
                    if step.done:
                        break
        finally:
            self.agent.learning_enabled = learning_state 