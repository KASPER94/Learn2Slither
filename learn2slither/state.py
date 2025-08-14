"""Encodeur d'état et utilitaires de diagnostic.

Etat minimal: (danger_ahead, danger_left, danger_right, dir_idx, food_dir)
- danger_* ∈ {0,1}
- dir_idx ∈ {0..3} pour (UP, RIGHT, DOWN, LEFT)
- food_dir ∈ {0..3} pour (front, left, right, back) relativement à la direction

Encodage final en entier: on compacte ces catégories dans une base mixte.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Coord = Tuple[int, int]

_DIR_TO_IDX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
_IDX_TO_DIR = {v: k for k, v in _DIR_TO_IDX.items()}


def _ahead_cell(head: Coord, direction: str) -> Coord:
    x, y = head
    if direction == "UP":
        return (x - 1, y)
    if direction == "DOWN":
        return (x + 1, y)
    if direction == "RIGHT":
        return (x, y + 1)
    return (x, y - 1)


def _left_dir(direction: str) -> str:
    return {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}[direction]


def _right_dir(direction: str) -> str:
    return {"LEFT": "UP", "DOWN": "LEFT", "RIGHT": "DOWN", "UP": "RIGHT"}[direction]


def _collision(point: Coord, size: int, body: List[Coord]) -> bool:
    x, y = point
    if x < 0 or x >= size or y < 0 or y >= size:
        return True
    # collision avec le corps (sauf tête)
    return point in body[1:]


def _food_relative_dir(head: Coord, food: Coord, direction: str) -> int:
    """Retourne la direction relative de la nourriture par rapport à la tête.
    0: front, 1: left, 2: right, 3: back
    """
    hx, hy = head
    fx, fy = food
    # vecteur vers food
    dx = fx - hx
    dy = fy - hy
    if direction == "UP":
        forward = (-1, 0)
        left = (0, -1)
        right = (0, 1)
    elif direction == "DOWN":
        forward = (1, 0)
        left = (0, 1)
        right = (0, -1)
    elif direction == "RIGHT":
        forward = (0, 1)
        left = (-1, 0)
        right = (1, 0)
    else:  # LEFT
        forward = (0, -1)
        left = (1, 0)
        right = (-1, 0)

    # produit scalaire signe pour décider de l'orientation principale
    def proj(v):
        return v[0] * dx + v[1] * dy

    pf, pl, pr = proj(forward), proj(left), proj(right)
    m = max(pf, pl, pr, key=lambda v: abs(v))
    if abs(m) == abs(pf):
        return 0 if pf >= 0 else 3
    if abs(m) == abs(pl):
        return 1 if pl >= 0 else 2
    return 2 if pr >= 0 else 1


def encode_state(env_obs: Dict) -> int:
    size = int(env_obs["size"])
    body: List[Coord] = list(env_obs["snake"])  # data from SnakeEnv.observation()
    head = body[0]
    direction: str = env_obs["dir"]

    # dangers
    ahead = _ahead_cell(head, direction)
    left = _ahead_cell(head, _left_dir(direction))
    right = _ahead_cell(head, _right_dir(direction))

    d_ahead = int(_collision(ahead, size, body))
    d_left = int(_collision(left, size, body))
    d_right = int(_collision(right, size, body))

    # nourriture: choisir la verte la plus proche (sinon la rouge)
    greens: List[Coord] = list(env_obs.get("green_apples", []))
    if greens:
        # distance Manhattan
        greens.sort(key=lambda g: abs(g[0] - head[0]) + abs(g[1] - head[1]))
        target = greens[0]
    else:
        target = env_obs.get("red_apples", head)

    food_dir = _food_relative_dir(head, target, direction)  # 0..3
    dir_idx = _DIR_TO_IDX[direction]  # 0..3

    # compactage: [danger bits]*  + dir_idx*4 + food_dir
    # 2^3 * 4 * 4 = 128 états
    state = (((d_ahead << 1) | d_left) << 1 | d_right)  # 0..7
    state = state * 4 + dir_idx  # 0..31
    state = state * 4 + food_dir  # 0..127
    return state


def nearest_green(head: Coord, apples: List[Coord]) -> Coord:
    """Retourne la pomme verte la plus proche de la tête (squelette)."""
    # TODO: implémenter une vraie distance (Manhattan)
    return apples[0] if apples else head 