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


# --- Mode d'entraînement externe minimal ---

def train_external_dqn() -> None:
    """Lance l'entraînement du DQN externe fourni.
    Utilise learn2slither/DQNAgent/dqn_agent.py::train().
    """
    # Import local pour éviter les problèmes d'import si non utilisé
    from learn2slither.DQNAgent.dqn_agent import train as external_train
    external_train()


def state_to_vector(state_int: int) -> np.ndarray:
    """Convertit l'état entier en vecteur one-hot de 128 dimensions."""
    vector = np.zeros(128, dtype=np.float32)
    vector[state_int] = 1.0
    return vector


def train_dqn(episodes: int, max_steps: int, verbose: bool = False, 
              save_model: str = None, load_model: str = None, save_every: int = 0):
    """Entraîne un agent DQN."""
    print(f"Démarrage entraînement DQN: {episodes} épisodes, {max_steps} steps max")
    
    # Lazy imports to avoid importing torch when not needed
    from learn2slither.DQNAgent.dqn_trainer import DQNTrainer, DQNTrainConfig

    cfg = DQNTrainConfig(
        episodes=episodes,
        max_steps=max_steps,
        verbose=verbose,
        save_model_path=save_model,
        load_model_path=load_model,
        save_every=save_every,
    )
    
    trainer = DQNTrainer()
    trainer.train(cfg)
    
    print("Entraînement DQN terminé!")


def train_dqn_visual(episodes: int, max_steps: int, verbose: bool = False, 
                     save_model: str = None, load_model: str = None, save_every: int = 0):
    """Entraîne un agent DQN avec affichage visuel en temps réel."""
    print(f"🎮 Démarrage entraînement DQN VISUEL simplifié:")
    print(f"   - {episodes} épisodes, {max_steps} steps max")
    print("   - Apprentissage immédiat comme Q-table")
    print("Contrôles: ESC=quitter, SPACE=pause, +/-=vitesse")
    
    setup_sdl_for_play()
    pg.init()
    
    env = SnakeEnv(GRID_SIZE)
    # Hyperparamètres identiques à la Q-table
    from learn2slither.DQNAgent.dqn_agent import DQNAgent
    agent = DQNAgent(
        state_size=128, 
        action_size=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )
    
    # Charge un modèle existant si fourni
    if load_model:
        try:
            agent.load(load_model)
            print(f"✅ Modèle chargé: {load_model}")
        except Exception as e:
            print(f"⚠️  Impossible de charger le modèle: {e}")
    
    window_size = GRID_SIZE * CELL_SIZE
    screen = pg.display.set_mode((window_size, window_size))
    pg.display.set_caption("Learn2Slither - DQN Simplifié")
    clock = pg.time.Clock()
    
    # Variables d'affichage
    running = True
    paused = False
    fps = 10  # Vitesse d'affichage
    max_length_seen = 0
    
    # Boucle d'entraînement visuelle
    for episode in range(1, episodes + 1):
        env.reset()
        s_int = encode_state(env.observation())
        s_vector = state_to_vector(s_int)
        episode_reward = 0.0
        steps = 0
        
        for step_num in range(max_steps):
            # Gestion des événements
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_SPACE:
                        paused = not paused
                    elif event.key == pg.K_PLUS or event.key == pg.K_EQUALS:
                        fps = min(fps + 2, 30)
                        print(f"Vitesse: {fps} FPS")
                    elif event.key == pg.K_MINUS:
                        fps = max(fps - 2, 1)
                        print(f"Vitesse: {fps} FPS")
            
            if not running:
                break
                
            if not paused:
                # Agent choisit une action
                a = agent.choose_action(s_vector)
                step_result = env.step(a)
                s2_int = encode_state(env.observation())
                s2_vector = state_to_vector(s2_int)
                
                # Calcule la récompense
                r = compute_reward(
                    died=step_result.done,
                    ate_green=step_result.grew,
                    ate_red=step_result.shrank,
                )
                episode_reward += r
                
                # Apprentissage immédiat comme Q-table
                agent.update(s_vector, a, r, s2_vector)
                
                s_vector = s2_vector
                steps += 1
                
                if step_result.done:
                    break
            
            # Affichage
            draw_scene(screen, env, CELL_SIZE)
            
            # Affiche les infos d'entraînement
            font = pg.font.Font(None, 24)
            info_lines = [
                f"Episode: {episode}/{episodes}",
                f"Steps: {steps}/{max_steps}",
                f"Score: {env.score}",
                f"Length: {len(env.snake)}",
                f"Max Length: {max(max_length_seen, len(env.snake))}",
                f"Reward: {episode_reward:.1f}",
                f"Epsilon: {agent.epsilon:.3f}",
                f"Learning: Immediate (like Q-table)"
            ]
            
            for i, line in enumerate(info_lines):
                color = (255, 255, 255) if not paused else (255, 255, 0)
                text = font.render(line, True, color)
                screen.blit(text, (10, 10 + i * 25))
            
            if paused:
                pause_text = font.render("PAUSE - Appuyez SPACE pour continuer", True, (255, 255, 0))
                screen.blit(pause_text, (10, window_size - 30))
            
            pg.display.flip()
            clock.tick(fps)
        
        if not running:
            break
        
        # Fin d'épisode
        agent.end_episode()
        current_length = len(env.snake)
        if current_length > max_length_seen:
            max_length_seen = current_length
        
        if verbose:
            print(f"Episode {episode:4d} | steps={steps:3d} | score={env.score:3d} | "
                  f"length={current_length:2d} | max_length={max_length_seen:2d} | "
                  f"return={episode_reward:7.2f} | eps={agent.epsilon:.3f}")
        
        # Sauvegarde périodique
        if save_model and save_every > 0 and episode % save_every == 0:
            agent.save(save_model)
            print(f"💾 Modèle sauvegardé: {save_model}")
    
    # Sauvegarde finale
    if save_model and running:
        agent.save(save_model)
        print(f"💾 Entraînement terminé - Modèle sauvegardé: {save_model}")
    
    pg.quit()
    print("🏁 Entraînement visuel terminé!")


def evaluate_visual(load_model: str | None) -> None:
    """Évalue visuellement un agent DQN."""
    print("Évaluation visuelle DQN...")
    if load_model is None:
        raise ValueError("Un modèle doit être chargé pour l'évaluation visuelle")
    setup_sdl_for_play()
    pg.init()

    env = SnakeEnv(GRID_SIZE)
    from learn2slither.DQNAgent.dqn_agent import DQNAgent
    agent = DQNAgent()
    print(f"Chargement du modèle: {load_model}")
    agent.load(load_model)
    agent.epsilon = 0.0  # Pas d'exploration

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
            state = agent.get_state(env)
            action = agent.get_action(state)
            reward, done, score = env.step_dqn(action)
            # agent.train_short_memory(state, action, reward, next_state, done)
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
    elif args.train_visual:
        train_dqn_visual(
            args.episodes,
            args.max_steps,
            verbose=args.verbose,
            save_model=args.save_model,
            load_model=args.load_model,
            save_every=args.save_every,
        )
    elif args.train:
        train_dqn(
            args.episodes,
            args.max_steps,
            verbose=args.verbose,
            save_model=args.save_model,
            load_model=args.load_model,
            save_every=args.save_every,
        )
    elif args.train_external:
        train_external_dqn()
    else:
        print("Utilisez --train pour démarrer l'entraînement DQN headless")
        print("Utilisez --train-visual pour voir l'entraînement en temps réel")
        print("Utilisez --vision pour évaluer un modèle entraîné")
        print()
        print("Exemples:")
        print("  python test_dqn.py --train-visual --episodes 100 --verbose --save-model models/dqn_simple.pth")
        print("  python test_dqn.py --vision --load-model models/dqn_simple.pth") 