#!/usr/bin/env python3
"""Script de test pour entraîner le DQN Agent."""

import argparse
from typing import Any  # lazy import DQNTrainer/DQNTrainConfig inside train_dqn
import pygame as pg
from learn2slither.env import SnakeEnv
from learn2slither.state import encode_state
from learn2slither.render import draw_scene
from learn2slither.controls import handle_events_for_manual, relative_action
# from learn2slither.DQNAgent.dqn_trainer import DQNTrainer  # lazy import where needed
from learn2slither.config import GRID_SIZE, CELL_SIZE
# lazy import DQNAgent inside train_dqn_visual and evaluate_visual
from learn2slither.rewards import compute_reward
import numpy as np
import os
import sys


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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test-dqn")
    parser.add_argument("--train", action="store_true", help="mode entraînement headless")
    parser.add_argument("--train-visual", action="store_true", help="mode entraînement avec affichage visuel")
    parser.add_argument("--train-external", action="store_true", help="entraîne le DQN externe (learn2slither/DQNAgent/dqn_agent.py)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", help="logs par épisode")
    parser.add_argument("--save-model", type=str, default=None, help="chemin .pth pour sauvegarder")
    parser.add_argument("--load-model", type=str, default=None, help="chemin .pth à charger")
    parser.add_argument("--save-every", type=int, default=0, help="sauvegarde toutes les N itérations")
    parser.add_argument("--vision", action="store_true", help="évalue visuellement un agent (utilise --load-model si fourni)")
    return parser


def evaluate_visual(load_model: str | None) -> None:
    """Évalue visuellement un agent DQN."""
    print("Évaluation visuelle DQN...")
    if load_model is None:
        raise ValueError("Un modèle doit être chargé pour l'évaluation visuelle")
    setup_sdl_for_play()
    pg.init()

    env = SnakeEnv(GRID_SIZE)
    from learn2slither.DQNAgent.model import Inference
    inference = Inference(load_model, 11, 256, 3)

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
            print(env.observation())
            print(env)
            state = inference.get_state(env)
            print(state)
            action = inference.get_action(state)
            # action_vector = [0, 0, 0]
            # action_vector[action] = 1
            _, done, _ = env.step_dqn(action)
            if done:
                env.reset()
        # Rendu
        draw_scene(screen, env, CELL_SIZE)
        pg.display.flip()
        clock.tick(8) 
    pg.quit()




if __name__ == "__main__":
    args = build_parser().parse_args()
    
    if args.vision:
        evaluate_visual(args.load_model)
    elif args.train:
        from learn2slither.DQNAgent.dqn_agent import train as train_dqn
        train_dqn()
    else:
        print("Utilisez --train pour démarrer l'entraînement DQN headless")
        print("Utilisez --train-visual pour voir l'entraînement en temps réel")
        print("Utilisez --vision pour évaluer un modèle entraîné")
        print()
        print("Exemples:")
        print("  python test_dqn.py --train-visual --episodes 100 --verbose --save-model models/dqn_simple.pth")
        print("  python test_dqn.py --vision --load-model models/dqn_simple.pth") 