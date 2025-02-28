import numpy as np
import pygame as pg
import random

GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

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
            if green_apples != red_apples and green_apples not in self.snake and red_apples not in self.snake:
                self.green_apples = green_apples
                self.red_apples = red_apples
                break

        self.board[self.green_apples] = 2
        self.board[self.red_apples] = 3



if __name__ == "__main":
    snakeEnv = SnakeEnv(10)
    pg.init()
    screen = pg.display.set_mode((WINDOW_SIZE * WINDOW_SIZE))
    clock = pg.time.Clock()
    running = True

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        screen.fill(WHITE)
        for i in range(0, WINDOW_SIZE, CELL_SIZE):
            for y in range(0, WINDOW_SIZE, CELL_SIZE):
                rect = pg.draw.Rect(i, y, CELL_SIZE, CELL_SIZE)
                pg.draw.rect(screen, BLACK, rect, 1)
        
        pg.draw.rect(screen, GREEN, (snakeEnv.green_apples[1] * CELL_SIZE, snakeEnv.green_apples[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pg.draw.rect(screen, RED, (snakeEnv.red_apples[1] * CELL_SIZE, snakeEnv.red_apples[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        clock.tick(5)