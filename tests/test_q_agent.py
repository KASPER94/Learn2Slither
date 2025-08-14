import math

from learn2slither.q_agent import QAgent
from learn2slither.actions import Action


def test_q_update_increases_towards_target():
    agent = QAgent(alpha=0.5, gamma=1.0, epsilon=0.0)
    s, a, r, s2 = 0, Action.STRAIGHT, 1.0, 1
    # force next state's best value
    agent.Q[s2][Action.STRAIGHT.value] = 3.0
    before = agent.Q[s][a.value]
    agent.update(s, a, r, s2)
    after = agent.Q[s][a.value]
    # target = r + gamma * max_next = 1 + 3 = 4
    # new = before + alpha*(target-before) = 0 + 0.5*4 = 2
    assert math.isclose(after, 2.0, rel_tol=1e-6)


def test_epsilon_decay_bounds():
    agent = QAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent.end_episode()
    assert agent.epsilon == 0.5
    # decay below min caps at epsilon_min
    agent.end_episode()
    assert agent.epsilon == 0.25
    agent.end_episode()
    assert agent.epsilon == 0.125
    agent.end_episode()
    assert agent.epsilon == 0.1 