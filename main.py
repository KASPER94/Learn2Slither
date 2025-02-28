import numpy as np
import random

class SnakeEnv:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        """Initialize new Snake Environmment"""
        self.board = np.zeros((self.size, self.size))
        self.snake = [(random.randint(0, self.size - 1), random.randint(0, self.size - 1))]
        self.board[self.snake[0]]
        self.spawn_apples()

    def spawn_apples(self):
        """Add apples on Snake Env"""
        while True:
            green_apples = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            red_apples = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if green_apples != red_apples and green_apples not int self.snake 