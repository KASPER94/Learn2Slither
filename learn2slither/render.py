"""Rendu Pygame pour Learn2Slither (facteurisé).

Fourni des fonctions utilitaires pour dessiner la grille et l'état de
l'environnement sur une surface Pygame.
"""

from __future__ import annotations

import pygame as pg

from .config import CELL_SIZE, COLORS, GRID_SIZE
from .env import SnakeEnv


def draw_grid(screen: pg.Surface, window_size: int, cell_size: int = CELL_SIZE) -> None:
    for i in range(0, window_size, cell_size):
        for y in range(0, window_size, cell_size):
            rect = pg.Rect(i, y, cell_size, cell_size)
            pg.draw.rect(screen, COLORS["WHITE"], rect, 1)


def draw_scene(screen: pg.Surface, env: SnakeEnv, cell_size: int = CELL_SIZE) -> None:
    screen.fill(COLORS["BLACK"])  # fond
    window_size = env.size * cell_size

    draw_grid(screen, window_size, cell_size)

    # pommes vertes
    for apple in env.green_apples:
        pg.draw.rect(
            screen,
            COLORS["GREEN"],
            (apple[1] * cell_size, apple[0] * cell_size, cell_size, cell_size),
        )

    # pomme rouge
    if env.red_apples is not None:
        pg.draw.rect(
            screen,
            COLORS["RED"],
            (
                env.red_apples[1] * cell_size,
                env.red_apples[0] * cell_size,
                cell_size,
                cell_size,
            ),
        )

    # serpent
    for seg in env.snake:
        pg.draw.rect(
            screen,
            COLORS["BLUE"],
            (seg[1] * cell_size, seg[0] * cell_size, cell_size, cell_size),
        ) 