"""CLI (squelette) pour entraîner/évaluer l'agent Learn2Slither.

Utilisation typique (une fois l'environnement migré):
    uv run python -m learn2slither.cli train --episodes 1000 --headless
"""

from __future__ import annotations

import argparse

from .q_agent import QAgent
from .trainer import Trainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learn2slither")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="entraîner un agent (squelette)")
    p_train.add_argument("--episodes", type=int, default=100)
    p_train.add_argument("--max-steps", type=int, default=500)
    p_train.add_argument("--headless", action="store_true")

    p_eval = sub.add_parser("eval", help="évaluer un agent (squelette)")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--max-steps", type=int, default=500)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # TODO: instancier l'env migré depuis main.py
    env = None  # placeholder
    agent = QAgent()
    trainer = Trainer(env, agent)

    if args.command == "train":
        cfg = TrainConfig(episodes=args.episodes, max_steps=args.max_steps, headless=args.headless)
        trainer.train(cfg)
    else:
        trainer.evaluate(episodes=args.episodes, max_steps=args.max_steps)

    return 0 