"""Fonctions de récompense pour le Snake RL.

Contient une fonction principale `compute_reward` configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    death: float = -100.0
    green: float = 25.0
    red: float = -25.0
    step: float = -0.25


def compute_reward(*, died: bool, ate_green: bool, ate_red: bool) -> float:
    """Calcule la récompense d'un pas (squelette simple)."""
    cfg = RewardConfig()
    if died:
        return cfg.death
    value = cfg.step
    if ate_green:
        value += cfg.green
    if ate_red:
        value += cfg.red
    return value 