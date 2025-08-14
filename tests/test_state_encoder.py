from learn2slither.state import encode_state


def make_obs(dir: str, snake, green=None, red=None, size=5):
    return {
        "dir": dir,
        "snake": snake,
        "green_apples": green or [],
        "red_apples": red,
        "size": size,
    }


def test_danger_ahead_on_wall():
    # head at top row facing up -> danger ahead
    obs = make_obs("UP", snake=[(0, 2), (1, 2), (2, 2)], green=[(4, 4)], size=5)
    state = encode_state(obs)
    # top 3 bits encode danger_ahead,left,right; ensure non-zero
    assert state // (4 * 4) in range(4, 8)  # at least one danger bit set, ahead should be 1


def test_food_direction_front():
    # food directly in front
    obs = make_obs("RIGHT", snake=[(2, 2), (2, 1), (2, 0)], green=[(2, 4)], size=5)
    state = encode_state(obs)
    # last 2 bits (mod 4) are food_dir; 0 => front
    assert state % 4 == 0 