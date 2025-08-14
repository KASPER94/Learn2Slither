"""Point d'entrée Learn2Slither.

Deux modes disponibles:
- mode jeu (par défaut): affiche le plateau avec Pygame et permet de jouer
- mode entraînement: lance la boucle RL (headless) avec Q-learning

Exemples:
    uv run python main.py                  # jeu visuel
    uv run python main.py --train --episodes 500  # entraînement headless
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pygame as pg

from learn2slither.actions import Action, Direction, next_direction
from learn2slither.config import CELL_SIZE, COLORS, GRID_SIZE
from learn2slither.env import SnakeEnv
from learn2slither.sdl import configure_sdl
from learn2slither.trainer import TrainConfig, Trainer
from learn2slither.q_agent import QAgent
from learn2slither.state import encode_state
from learn2slither.render import draw_scene
from learn2slither.controls import handle_events_for_manual, relative_action

# ---------------------------------------------------------------------------
# Utilitaires SDL + Pygame
# ---------------------------------------------------------------------------


def setup_sdl_for_play() -> None:
    """Configure SDL pour un affichage natif selon l'OS (jeu visuel)."""
    if "SDL_VIDEODRIVER" not in os.environ:
        if sys.platform.startswith("darwin"):
            os.environ["SDL_VIDEODRIVER"] = "cocoa"
        elif sys.platform.startswith("linux"):
            os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")


# ---------------------------------------------------------------------------
# Mode JEU (visuel)
# ---------------------------------------------------------------------------


def _relative_action(current: Direction, desired: Direction) -> Action:
    """Calcule l'action relative (gauche/avant/droite) pour aller vers `desired`."""
    if desired == current:
        return Action.STRAIGHT
    if next_direction(current, Action.TURN_LEFT) == desired:
        return Action.TURN_LEFT
    return Action.TURN_RIGHT


def play() -> None:
    """Boucle de jeu visuelle avec Pygame (aucun apprentissage)."""
    setup_sdl_for_play()
    pg.init()

    env = SnakeEnv(GRID_SIZE)  # data from learn2slither.env
    window_size = GRID_SIZE * CELL_SIZE
    screen = pg.display.set_mode((window_size, window_size))
    pg.display.set_caption("Learn2Slither - Snake RL")
    clock = pg.time.Clock()

    running = True
    paused = False
    desired_dir: Optional[Direction] = None

    while running:
        action, quit_req, toggle_pause = handle_events_for_manual(env)
        if quit_req:
            running = False
        if toggle_pause:
            paused = not paused

        if not paused:
            if action is None:
                action = Action.STRAIGHT
            step = env.step(action)
            if step.done:
                env.reset()

        # Rendu --------------------------------------------------------------
        draw_scene(screen, env, CELL_SIZE)
        pg.display.flip()
        clock.tick(5)

    pg.quit()


# ---------------------------------------------------------------------------
# Mode ENTRAÎNEMENT (headless)
# ---------------------------------------------------------------------------


def train(episodes: int, max_steps: int, *, verbose: bool, save_model: str | None, load_model: str | None, save_every: int) -> None:
    """Lance un entraînement (headless) avec options de log et de sauvegarde."""
    configure_sdl(headless=True)  # pas d'affichage pendant l'entraînement
    trainer = Trainer()  # crée un SnakeEnv interne par défaut
    cfg = TrainConfig(
        episodes=episodes,
        max_steps=max_steps,
        headless=True,
        verbose=verbose,
        save_model_path=save_model,
        load_model_path=load_model,
        save_every=save_every,
    )
    trainer.train(cfg)


def evaluate_visual(load_model: str | None) -> None:
    """Affiche une session où l'agent (Q-table) joue visuellement.

    Si `load_model` est fourni, on charge la Q-table; sinon on démarre avec
    une table vide (peu performant au début).
    """
    setup_sdl_for_play()
    pg.init()

    env = SnakeEnv(GRID_SIZE)
    agent = QAgent()
    if load_model:
        agent = QAgent.load(load_model)
    agent.learning_enabled = False  # exploitation uniquement

    window_size = GRID_SIZE * CELL_SIZE
    screen = pg.display.set_mode((window_size, window_size))
    pg.display.set_caption("Learn2Slither - Agent Vision")
    clock = pg.time.Clock()

    running = True
    paused = False
    while running:
        # Utilise le handler commun pour capter ESC/SPACE/QUIT
        _, quit_req, toggle_pause = handle_events_for_manual(env)
        if quit_req:
            running = False
        if toggle_pause:
            paused = not paused

        if not paused:
            # agent choisit une action à partir de l'état encodé
            s = encode_state(env.observation())
            a = agent.choose_action(s)
            step = env.step(a)
            if step.done:
                env.reset()

        # Rendu
        draw_scene(screen, env, CELL_SIZE)
        pg.display.flip()
        clock.tick(8)  # un peu plus rapide que le mode manuel

    pg.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learn2slither-main")
    parser.add_argument("--train", action="store_true", help="mode entraînement headless")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--verbose", action="store_true", help="logs par épisode")
    parser.add_argument("--save-model", type=str, default=None, help="chemin JSON pour sauvegarder la Q-table")
    parser.add_argument("--load-model", type=str, default=None, help="chemin JSON à charger avant l'entraînement")
    parser.add_argument("--save-every", type=int, default=0, help="sauvegarde toutes les N itérations (0 = fin seulement)")
    parser.add_argument("--vision", action="store_true", help="évalue visuellement un agent (utilise --load-model si fourni)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.vision:
        evaluate_visual(args.load_model)
    elif args.train:
        train(
            args.episodes,
            args.max_steps,
            verbose=args.verbose,
            save_model=args.save_model,
            load_model=args.load_model,
            save_every=args.save_every,
        )
    else:
        play()
