#!/usr/bin/env python3
"""Script de test pour entraîner le DQN Agent."""

import argparse
from learn2slither.dqn_trainer import DQNTrainer, DQNTrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test-dqn")
    parser.add_argument("--train", action="store_true", help="mode entraînement headless")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", help="logs par épisode")
    parser.add_argument("--save-model", type=str, default=None, help="chemin .pth pour sauvegarder")
    parser.add_argument("--load-model", type=str, default=None, help="chemin .pth à charger")
    parser.add_argument("--save-every", type=int, default=0, help="sauvegarde toutes les N itérations")
    return parser


def train_dqn(episodes: int, max_steps: int, verbose: bool = False, 
              save_model: str = None, load_model: str = None, save_every: int = 0):
    """Entraîne un agent DQN."""
    print(f"Démarrage entraînement DQN: {episodes} épisodes, {max_steps} steps max")
    
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


if __name__ == "__main__":
    args = build_parser().parse_args()
    
    if args.train:
        train_dqn(
            args.episodes,
            args.max_steps,
            verbose=args.verbose,
            save_model=args.save_model,
            load_model=args.load_model,
            save_every=args.save_every,
        )
    else:
        print("Utilisez --train pour démarrer l'entraînement DQN")
        print("Exemple: python test_dqn.py --train --episodes 100 --verbose --save-model models/dqn_model.pth") 