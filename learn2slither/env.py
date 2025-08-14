"""Environnement Snake (logique pure, sans Pygame).

Migration de la logique depuis `main.py`. Cette version n'effectue aucun rendu et
se contente de gérer l'état du jeu et les transitions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .actions import Action, next_direction, Direction

Coord = Tuple[int, int]


@dataclass
class StepResult:
    """Résultat d'un pas d'environnement."""
    done: bool
    grew: bool
    shrank: bool


class SnakeEnv:
    """Environnement Snake sans dépendance à Pygame.

    Les données proviennent de la version de référence présente dans `main.py`.
    """

    def __init__(self, size: int = 10):
        # données de source: `main.py` (logique actuelle)
        self.size: int = size
        self.dir: Direction = "RIGHT"
        self.snake: List[Coord] = []
        self.green_apples: List[Coord] = []
        self.red_apples: Optional[Coord] = None
        self.score: int = 0
        self.reset()

    # ------------------------------ Logique core ---------------------------
    def reset(self) -> None:
        """Réinitialise l'environnement (placement serpent + pommes)."""
        self.snake = []
        self.score = 0

        orientation = random.choice(["HORIZONTAL", "VERTICAL", "L"])

        if orientation == "HORIZONTAL":
            self.dir = random.choice(["LEFT", "RIGHT"])  # type: ignore[assignment]
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)

            if self.dir == "LEFT":
                # Pour direction LEFT, la tête doit être à gauche
                self.snake = [
                    (start_x, start_y),
                    (start_x, start_y + 1),
                    (start_x, start_y + 2),
                ]
            else:  # RIGHT
                # Pour direction RIGHT, la tête doit être à droite
                self.snake = [
                    (start_x, start_y),
                    (start_x, start_y - 1),
                    (start_x, start_y - 2),
                ]

        elif orientation == "VERTICAL":
            self.dir = random.choice(["UP", "DOWN"])  # type: ignore[assignment]
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)

            if self.dir == "UP":
                # Tête en haut
                self.snake = [
                    (start_x, start_y),
                    (start_x + 1, start_y),
                    (start_x + 2, start_y),
                ]
            else:  # DOWN
                # Tête en bas
                self.snake = [
                    (start_x, start_y),
                    (start_x - 1, start_y),
                    (start_x - 2, start_y),
                ]

        else:  # Cas "L"
            self.dir = random.choice(["UP", "LEFT", "RIGHT", "DOWN"])  # type: ignore[assignment]
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)

            if self.dir == "UP":
                self.snake = [
                    (start_x, start_y),
                    (start_x + 1, start_y),
                    (start_x + 1, start_y + 1),
                ]
            elif self.dir == "LEFT":
                self.snake = [
                    (start_x, start_y),
                    (start_x, start_y + 1),
                    (start_x + 1, start_y + 1),
                ]
            elif self.dir == "RIGHT":
                self.snake = [
                    (start_x, start_y),
                    (start_x, start_y - 1),
                    (start_x + 1, start_y - 1),
                ]
            else:  # DOWN
                self.snake = [
                    (start_x, start_y),
                    (start_x - 1, start_y),
                    (start_x - 1, start_y - 1),
                ]

        # Validation et respawn si nécessaire
        if not self._validate_snake():
            self.reset()
            return

        self._spawn_green_apples()
        self._spawn_red_apple()

    def _validate_snake(self) -> bool:
        """Vérifie que le serpent est entièrement dans les limites et sans doublons."""
        for x, y in self.snake:
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                return False
        return len(self.snake) == len(set(self.snake))

    def _spawn_green_apples(self) -> None:
        """Ajoute deux pommes vertes sur le plateau."""
        if self.green_apples is None:
            self.green_apples = []
        while len(self.green_apples) != 2:
            apple = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if apple not in self.snake and apple not in self.green_apples:
                self.green_apples.append(apple)

    def _spawn_red_apple(self) -> None:
        """Ajoute une pomme rouge unique sur le plateau."""
        while True:
            red = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if red not in self.snake and red not in self.green_apples:
                self.red_apples = red
                break

    def _collision(self, point: Coord) -> bool:
        """Collision mur ou corps (hors nouvelle tête)."""
        x, y = point
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        for i in range(1, len(self.snake)):
            if point == self.snake[i]:
                return True
        return False

    def _update_snake_size(self, *, grow: bool = False, shrink: bool = False) -> bool:
        """Gère la taille du serpent après un mouvement.

        Retourne False si le rétrécissement n'est pas possible (game over).
        """
        if grow:
            return True
        if shrink:
            if len(self.snake) > 1:
                self.snake.pop()
                if len(self.snake) > 1:
                    self.snake.pop()
            else:
                return False
        else:
            self.snake.pop()
        return True

    def _advance(self) -> Optional[str]:
        """Fait avancer le serpent d'une case.

        Retourne "green" si une pomme verte est mangée, "red" si la rouge est
        mangée, None sinon. Retourne None et l'appelant doit vérifier la collision
        préalable.
        """
        head_x, head_y = self.snake[0]
        if self.dir == "UP":
            new_head = (head_x - 1, head_y)
        elif self.dir == "DOWN":
            new_head = (head_x + 1, head_y)
        elif self.dir == "RIGHT":
            new_head = (head_x, head_y + 1)
        else:  # LEFT
            new_head = (head_x, head_y - 1)

        if self._collision(new_head):
            return "dead"

        self.snake.insert(0, new_head)

        if new_head in self.green_apples:
            self.green_apples.remove(new_head)
            self._spawn_green_apples()
            return "green"
        if self.red_apples is not None and new_head == self.red_apples:
            self._spawn_red_apple()
            return "red"
        return None

    # ------------------------------- API RL --------------------------------
    def step(self, action: Action) -> StepResult:
        """Applique une action relative et fait un pas d'environnement.

        Paramètres
        ----------
        action: Action
            Action relative à la direction courante (gauche/avant/droite).
        """
        # Appliquer la nouvelle direction
        self.dir = next_direction(self.dir, action)
        ate = self._advance()
        if ate == "dead":
            return StepResult(done=True, grew=False, shrank=False)

        grew = False
        shrank = False
        if ate == "green":
            grew = True
            # croissance: on garde la queue (ne pas pop)
            self._update_snake_size(grow=True)
        elif ate == "red":
            shrank = True
            if not self._update_snake_size(shrink=True):
                return StepResult(done=True, grew=False, shrank=True)
        else:
            # mouvement simple
            self._update_snake_size()

        self.score = len(self.snake) - 3
        return StepResult(done=False, grew=grew, shrank=shrank)

    def observation(self):
        """Retourne une observation brute (positions, direction, taille)."""
        return {
            "dir": self.dir,
            "snake": list(self.snake),
            "green_apples": list(self.green_apples),
            "red_apples": self.red_apples,
            "size": self.size,
        } 