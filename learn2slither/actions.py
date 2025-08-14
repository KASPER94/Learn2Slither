"""Définition des actions et utilitaires de direction.

Les actions sont relatives à la direction courante du serpent afin de réduire
l'espace d'actions et de faciliter l'apprentissage.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

Direction = Literal["UP", "DOWN", "LEFT", "RIGHT"]


class Action(Enum):
    TURN_LEFT = 0
    STRAIGHT = 1
    TURN_RIGHT = 2


_LEFT_TURN = {
    "UP": "LEFT",
    "LEFT": "DOWN",
    "DOWN": "RIGHT",
    "RIGHT": "UP",
}

_RIGHT_TURN = {v: k for k, v in _LEFT_TURN.items()}


def next_direction(current: Direction, action: Action) -> Direction:
    """Retourne la prochaine direction en fonction de l'action.

    Cette fonction ne modifie pas l'état; elle calcule seulement la nouvelle
    direction à appliquer sur l'environnement.
    """
    if action == Action.STRAIGHT:
        return current
    if action == Action.TURN_LEFT:
        return _LEFT_TURN[current]
    return _RIGHT_TURN[current] 