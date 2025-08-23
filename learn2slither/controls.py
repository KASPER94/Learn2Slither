"""Contrôles clavier et conversion direction -> action.

- `handle_events_for_manual(env)` lit les événements Pygame et renvoie un
  triplet: (action_relative|None, quit_requis, toggle_pause).
"""

from __future__ import annotations

from typing import Optional, Tuple

import pygame as pg

from .actions import Action, Direction, next_direction
from .env import SnakeEnv
from collections import deque


def relative_action(current: Direction, desired: Direction) -> Action:
    if desired == current:
        return Action.STRAIGHT
    if next_direction(current, Action.TURN_LEFT) == desired:
        return Action.TURN_LEFT
    return Action.TURN_RIGHT

KEY_BUFFER = deque(maxlen=2)  # persistant au module

def handle_events_for_manual(env):
    desired_dir = None
    quit_requested = False
    toggle_pause = False

    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit_requested = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                quit_requested = True
            elif event.key == pg.K_SPACE:
                toggle_pause = True
            elif event.key in (pg.K_q, pg.K_z, pg.K_s, pg.K_d, pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT):
                KEY_BUFFER.append(event.key)

    # Combinaisons ou dernière touche valable
    if len(KEY_BUFFER) == 2:
        first, second = KEY_BUFFER[0], KEY_BUFFER[1]
        # à toi de définir une règle claire; sinon on garde la dernière touche:
        key = second
        KEY_BUFFER.clear()
    elif len(KEY_BUFFER) == 1:
        key = KEY_BUFFER[-1]
        KEY_BUFFER.clear()
    else:
        key = None

    if key == pg.K_z: desired_dir = "UP"
    elif key == pg.K_UP: desired_dir = "UP"
    elif key == pg.K_s: desired_dir = "DOWN"
    elif key == pg.K_DOWN: desired_dir = "DOWN"
    elif key == pg.K_q: desired_dir = "LEFT"
    elif key == pg.K_LEFT: desired_dir = "LEFT"
    elif key == pg.K_d: desired_dir = "RIGHT"
    elif key == pg.K_RIGHT: desired_dir = "RIGHT"

    action = relative_action(env.dir, desired_dir) if desired_dir else None
    return action, quit_requested, toggle_pause