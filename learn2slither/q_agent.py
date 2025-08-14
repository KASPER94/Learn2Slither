"""Agent Q-learning basé Q-table.

Implémentation minimale de la fonction Q sous forme de table. La politique est
ε-greedy et l'agent peut sauvegarder/charger son état.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from .actions import Action


class QAgent:
    """Agent Q-learning avec Q-table."""

    def __init__(
        self,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        learning_enabled: bool = True,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_enabled = learning_enabled

        # Q-table: mapping état -> liste de Q-valeurs par action
        self.Q: Dict[int, List[float]] = defaultdict(
            lambda: [0.0 for _ in range(self.n_actions)]
        )

    def choose_action(self, state: int) -> Action:
        """Choisit une action via ε-greedy."""
        from random import random, randint

        if random() < self.epsilon:
            return Action(randint(0, self.n_actions - 1))
        values = self.Q[state]
        best_index = max(range(self.n_actions), key=lambda i: values[i])
        return Action(best_index)

    def update(self, s: int, a: Action, r: float, s2: int) -> None:
        """Met à jour la Q-table selon la règle Q-learning."""
        if not self.learning_enabled:
            return
        q_sa = self.Q[s][a.value]
        max_next = max(self.Q[s2])
        td_target = r + self.gamma * max_next
        self.Q[s][a.value] = q_sa + self.alpha * (td_target - q_sa)

    def end_episode(self) -> None:
        """Décroît epsilon en fin d'épisode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        """Sauvegarde la Q-table sur disque (JSON compact)."""
        data = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_enabled": self.learning_enabled,
            "Q": {str(k): v for k, v in self.Q.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "QAgent":
        """Charge un agent depuis un fichier JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent = cls(
            n_actions=data.get("n_actions", 3),
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 1.0),
            epsilon_min=data.get("epsilon_min", 0.05),
            epsilon_decay=data.get("epsilon_decay", 0.995),
            learning_enabled=data.get("learning_enabled", True),
        )
        agent.Q = {int(k): list(map(float, v)) for k, v in data.get("Q", {}).items()}
        return agent 