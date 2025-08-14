"""Package Learn2Slither

Ce package contient l'environnement Snake, l'agent Q-learning et les utilitaires
nécessaires pour l'entraînement et l'évaluation.
"""

from .config import GRID_SIZE, CELL_SIZE, COLORS  # noqa: F401
from .actions import Action, next_direction  # noqa: F401
from .q_agent import QAgent  # noqa: F401 