"""Contrôles clavier et conversion direction -> action.

- `handle_events_for_manual(env)` lit les événements Pygame et renvoie un
  triplet: (action_relative|None, quit_requis, toggle_pause).
"""

from __future__ import annotations

from typing import Optional, Tuple

import pygame as pg

from .actions import Action, Direction, next_direction
from .env import SnakeEnv


def relative_action(current: Direction, desired: Direction) -> Action:
    if desired == current:
        return Action.STRAIGHT
    if next_direction(current, Action.TURN_LEFT) == desired:
        return Action.TURN_LEFT
    return Action.TURN_RIGHT


def handle_events_for_manual(env: SnakeEnv) -> Tuple[Optional[Action], bool, bool]:
    """Lit les événements clavier et renvoie (action, quit, toggle_pause)."""
    desired_dir: Optional[Direction] = None
    quit_requested = False
    toggle_pause = False

    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit_requested = True
        elif event.type == pg.KEYDOWN:
            if event.key in (pg.K_w, pg.K_UP) and env.dir != "DOWN":
                desired_dir = "UP"
            elif event.key in (pg.K_s, pg.K_DOWN) and env.dir != "UP":
                desired_dir = "DOWN"
            elif event.key in (pg.K_a, pg.K_LEFT) and env.dir != "RIGHT":
                desired_dir = "LEFT"
            elif event.key in (pg.K_d, pg.K_RIGHT) and env.dir != "LEFT":
                desired_dir = "RIGHT"
            elif event.key == pg.K_ESCAPE:
                quit_requested = True
            elif event.key == pg.K_SPACE:
                toggle_pause = True

    action: Optional[Action] = None
    if desired_dir is not None:
        action = relative_action(env.dir, desired_dir)
    return action, quit_requested, toggle_pause 