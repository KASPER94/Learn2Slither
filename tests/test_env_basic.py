from learn2slither.env import SnakeEnv
from learn2slither.actions import Action


def test_reset_creates_valid_snake():
    env = SnakeEnv(10)
    env.reset()
    assert len(env.snake) >= 3
    # all segments inside bounds and unique
    assert all(0 <= x < env.size and 0 <= y < env.size for x, y in env.snake)
    assert len(env.snake) == len(set(env.snake))
    # apples
    assert len(env.green_apples) == 2
    assert env.red_apples is not None


def test_step_progress_and_score():
    env = SnakeEnv(10)
    before_len = len(env.snake)
    res = env.step(Action.STRAIGHT)
    assert res is not None
    assert len(env.snake) >= 1
    # score is length - 3
    assert env.score == len(env.snake) - 3 