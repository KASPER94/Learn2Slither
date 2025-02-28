import numpy as np
import pygame as pg
import random
import os
os.environ["SDL_VIDEODRIVER"] = "x11"  
os.environ["SDL_RENDER_DRIVER"] = "software" 
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1" 
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"

GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeEnv:
    def __init__(self, size=GRID_SIZE):
        self.dir = "RIGTH"
        self.started = False
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

    def update_snake(self):
        """Move snake"""
        if not self.started or self.dir is None:
            return
        head_x, head_y = self.snake[0]
        new_head = (0, 0)
        if self.dir == "UP":
            new_head = (head_x - 1, head_y)
        elif self.dir == "DOWN":
            new_head = (head_x + 1, head_y)
        elif self.dir == "RIGHT":
            new_head = (head_x, head_y + 1)
        elif self.dir == "LEFT":
            new_head = (head_x, head_y - 1)

        if new_head in self.snake or new_head[0] < 0 or new_head[0] >= self.size or new_head[1] >= self.size:
            print("Game Over")
            self.reset()
            return
        
        self.snake.insert(0, new_head)

        if new_head == self.green_apples:
            self.spawn_apples()
        else:
            self.snake.pop()

def key_event():
    keys = pg.key.get_pressed()
    if keys[pg.K_w] or keys[pg.K_UP]:
        if snakeEnv.dir != "DOWN":
            snakeEnv.dir = "UP"
            snakeEnv.started = True
    elif keys[pg.K_s] or keys[pg.K_DOWN]:
        if snakeEnv.dir != "UP":
            snakeEnv.dir = "DOWN"
            snakeEnv.started = True
    elif keys[pg.K_a] or keys[pg.K_LEFT]:
        if snakeEnv.dir != "RIGHT":
            snakeEnv.dir = "LEFT"
            snakeEnv.started = True
    elif keys[pg.K_d] or keys[pg.K_RIGHT]:
        if snakeEnv.dir != "LEFT":
            snakeEnv.dir = "RIGHT"
            snakeEnv.started = True


if __name__ == "__main__":
    pg.init()
    snakeEnv = SnakeEnv(10)
    screen = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pg.display.set_caption("Learn2Slither - Snake RL")
    clock = pg.time.Clock()
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        key_event()
        snakeEnv.update_snake()

        screen.fill(BLACK)

        for i in range(0, WINDOW_SIZE, CELL_SIZE):
            for y in range(0, WINDOW_SIZE, CELL_SIZE):
                rect = pg.Rect(i, y, CELL_SIZE, CELL_SIZE)
                pg.draw.rect(screen, WHITE, rect, 1)
        
        pg.draw.rect(screen, GREEN, (snakeEnv.green_apples[1] * CELL_SIZE, snakeEnv.green_apples[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pg.draw.rect(screen, RED, (snakeEnv.red_apples[1] * CELL_SIZE, snakeEnv.red_apples[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for seg in snakeEnv.snake:
            pg.draw.rect(screen, BLUE, (seg[1] * CELL_SIZE, seg[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        

        pg.display.flip()
        clock.tick(5)
    pg.quit()
