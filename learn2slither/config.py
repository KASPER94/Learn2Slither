"""Configuration globale du projet Learn2Slither.

Toutes les constantes et hyperparamètres par défaut sont centralisés ici afin de
faciliter l'ajustement et l'expérimentation.
"""

# Tailles de la grille/affichage
GRID_SIZE: int = 10
CELL_SIZE: int = 100

# Couleurs utilisées par le rendu (si activé)
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "BLUE": (0, 0, 255),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
}

# Hyperparamètres Q-learning par défaut
ALPHA: float = 0.1  # taux d'apprentissage
GAMMA: float = 0.95  # facteur d'actualisation
EPSILON_START: float = 1.0
EPSILON_MIN: float = 0.05
EPSILON_DECAY: float = 0.995  # décroissance multiplicative par épisode 