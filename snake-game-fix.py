import numpy as np
import pygame as pg
import random
import os
# os.environ["SDL_VIDEODRIVER"] = "x11"  
os.environ["SDL_VIDEODRIVER"] = "cocoa"
os.environ["SDL_RENDER_DRIVER"] = "software" 
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1" 
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"

GRID_SIZE = 10
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeEnv:
    def __init__(self, size=GRID_SIZE):
        self.dir = "RIGHT"
        self.size = size
        self.green_apples = []
        self.reset()

    def reset(self):
        """Initialize new Snake Environment"""
        self.board = np.zeros((self.size, self.size), dtype = str)
        self.snake = []
        self.score = len(self.snake)
        orientation = random.choice(["HORIZONTAL", "VERTICAL", "L"])

        if orientation == "HORIZONTAL":
            self.dir = random.choice(["LEFT", "RIGHT"])
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)
        
            if self.dir == "LEFT":
                # Pour direction LEFT, la tête doit être à gauche
                self.snake = [(start_x, start_y), (start_x, start_y + 1), (start_x, start_y + 2)]
            else:  # RIGHT
                # Pour direction RIGHT, la tête doit être à droite
                self.snake = [(start_x, start_y), (start_x, start_y - 1), (start_x, start_y - 2)]
        
        elif orientation == "VERTICAL":
            self.dir = random.choice(["UP", "DOWN"])
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)
        
            if self.dir == "UP":
                # Pour direction UP, la tête doit être en haut
                self.snake = [(start_x, start_y), (start_x + 1, start_y), (start_x + 2, start_y)]
            else:  # DOWN
                # Pour direction DOWN, la tête doit être en bas
                self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        
        else:  # Cas "L"
            self.dir = random.choice(["UP", "LEFT", "RIGHT", "DOWN"])
            start_x = random.randint(2, self.size - 3)
            start_y = random.randint(2, self.size - 3)
        
            if self.dir == "UP":
                self.snake = [(start_x, start_y), (start_x + 1, start_y), (start_x + 1, start_y + 1)]
            elif self.dir == "LEFT":
                self.snake = [(start_x, start_y), (start_x, start_y + 1), (start_x + 1, start_y + 1)]
            elif self.dir == "RIGHT":
                self.snake = [(start_x, start_y), (start_x, start_y - 1), (start_x + 1, start_y - 1)]
            else:  # self.dir == "DOWN"
                self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 1, start_y - 1)]
                
        # Vérifier que tous les segments sont valides
        if not self.validate_snake():
            # print("Snake initialization invalid, retrying...")
            self.reset()
            return
            
        # print(f"Snake initialized: {self.snake}")
        # print(f"Initial direction: {self.dir}")
        
        self.spawn_apples()
        self.spawn_red_apples_only()

    def validate_snake(self):
        """Vérifier que le serpent est entièrement dans les limites"""
        for seg in self.snake:
            x, y = seg
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                return False
                
        # Vérifier qu'il n'y a pas de segments dupliqués
        return len(self.snake) == len(set(self.snake))

    def spawn_apples(self):
        """Add apples on Snake Env"""
        if self.green_apples == None:
            self.green_apples = []
        while len(self.green_apples) != 2:
            apple = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if apple not in self.snake and apple not in self.green_apples:
                self.green_apples.append(apple)
                
        # while True:
        #     red_apples = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        #     if red_apples not in self.snake and red_apples not in self.green_apples:
        #         self.red_apples = red_apples
        #         break
            
    def spawn_red_apples_only(self):
        """Add apples on Snake Env"""
        while True:
            red_apples = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if red_apples not in self.snake and red_apples not in self.green_apples:
                self.red_apples = red_apples
                break

    def collision(self, point):
        """Vérifie si `point` est en dehors de la grille ou si le serpent se mord"""
        x, y = point 

        # Vérifier les limites de la grille
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            # print(f"Wall collision at {point}")
            return True 

        # Vérifier les collisions avec le corps du serpent (sauf la queue qui va disparaître)
        # Si le snake doit grandir, la queue ne disparaîtra pas, donc il faut vérifier tout le corps
        for i in range(1, len(self.snake)):
            if point == self.snake[i]:
                # print(f"Self collision at {point} with segment {i}")
                return True

        return False  

    def update_snake_size(self, grow=False, shrink=False):
        """Gère la taille du serpent après un mouvement."""
        if grow:
            return True

        if shrink:
            if len(self.snake) > 1:
                self.snake.pop()
                if len(self.snake) > 1:
                    self.snake.pop()
            else:
                return False
        else:
            self.snake.pop()
        return True

    def update_snake(self):
        """Move snake"""
        if not self.snake:
            return False
            
        head_x, head_y = self.snake[0]
        new_head = None
        
        # Correction de la logique de direction
        if self.dir == "UP":
            new_head = (head_x - 1, head_y)  # Déplacement vers le haut = diminuer la coordonnée x
        elif self.dir == "DOWN":
            new_head = (head_x + 1, head_y)  # Déplacement vers le bas = augmenter la coordonnée x
        elif self.dir == "RIGHT":
            new_head = (head_x, head_y + 1)  # Déplacement à droite = augmenter la coordonnée y
        elif self.dir == "LEFT":
            new_head = (head_x, head_y - 1)  # Déplacement à gauche = diminuer la coordonnée y
        
        # print(f"Direction: {self.dir}, Head: {self.snake[0]}, New Head: {new_head}")

        if self.collision(new_head):
            # print(f"Game Over - Collision at {new_head}")
            return False
        
        # Insérer la nouvelle tête à la position 0
        self.snake.insert(0, new_head)

        if new_head in self.green_apples:
            self.green_apples.remove(new_head)
            self.update_snake_size(grow=True)
            # if len(self.green_apples) < 1:
            self.spawn_apples()
        elif new_head == self.red_apples:
            self.spawn_red_apples_only()
            return self.update_snake_size(shrink=True)
        else:
            self.update_snake_size()
        return True

def key_event(running, paused, snakeEnv):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            return False, paused
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_w or event.key == pg.K_UP:
                if snakeEnv.dir != "DOWN":
                    snakeEnv.dir = "UP"
            elif event.key == pg.K_s or event.key == pg.K_DOWN:
                if snakeEnv.dir != "UP":
                    snakeEnv.dir = "DOWN"
            elif event.key == pg.K_a or event.key == pg.K_LEFT:
                if snakeEnv.dir != "RIGHT":
                    snakeEnv.dir = "LEFT"
            elif event.key == pg.K_d or event.key == pg.K_RIGHT:
                if snakeEnv.dir != "LEFT":
                    snakeEnv.dir = "RIGHT"
            elif event.key == pg.K_ESCAPE:
                running = False
            elif event.key == pg.K_SPACE:
                paused = not paused

    return running, paused

def get_state(snake_env):
    head = snake_env.snake[0]
    head_x, head_y = head
    head_y += 1
    head_x += 1
    dir = snake_env.dir
    board = np.full((snake_env.size + 2, snake_env.size + 2), "0", dtype=str)
    board[0, :] = "W"
    board[:, 0] = "W"
    board[:, -1] = "W"
    board[-1, :] = "W"

    for i, segment in enumerate(snake_env.snake):
        x, y = segment
        x += 1
        y += 1
        if i == 0:
            board[x, y] = "H"
        else:
            board[x, y] = "S"
    
    for a in snake_env.green_apples:
        x, y = a
        x += 1
        y += 1
        board[x, y] = "G"

    x, y = snake_env.red_apples
    x += 1
    y += 1
    board[x, y] = "R"

    vision = np.full((snake_env.size + 2, snake_env.size + 2), " ", dtype=str)
    if dir == "UP":
        for y in range(head_y, -1, -1): 
            vision[head_x, y] = board[head_x, y]
        for y in range(head_y + 1, snake_env.size + 2, 1):
            vision[head_x, y] = board[head_x, y]
        for x in range(head_x - 1, -1, -1):
            vision[x, head_y] = board[x, head_y]
        for x in range(head_x + 1, snake_env.size + 2, 1):
            vision[x, head_y] = board[x, head_y]

    elif dir == "DOWN":
        for y in range(head_y, snake_env.size + 2, 1):
            vision[head_x, y] = board[head_x, y]
        for y in range(head_y - 1, -1, -1):
            vision[head_x, y] = board[head_x, y]
        for x in range(head_x - 1, -1, -1):
            vision[x, head_y] = board[x, head_y]
        for x in range(head_x + 1, snake_env.size + 2, 1):
            vision[x, head_y] = board[x, head_y]        

    elif dir == "LEFT":
        for x in range(head_x, -1, -1): 
            vision[x, head_y] = board[x, head_y]
        for x in range(head_x + 1, snake_env.size + 2, 1): 
            vision[x, head_y] = board[x, head_y]
        for y in range(head_y - 1, -1, -1):  
            vision[head_x, y] = board[head_x, y]
        for y in range(head_y + 1, snake_env.size + 2, 1):
            vision[head_x, y] = board[head_x, y]

    elif dir == "RIGHT":
        for x in range(head_x, snake_env.size + 2, 1):
            vision[x, head_y] = board[x, head_y]
        for x in range(head_x - 1, -1, -1):
            vision[x, head_y] = board[x, head_y]
        for y in range(head_y - 1, -1, -1):
            vision[head_x, y] = board[head_x, y]
        for y in range(head_y + 1, snake_env.size + 2, 1):
            vision[head_x, y] = board[head_x, y]

    x, y = vision.shape
    for i in range(x):
        for j in range(y):
            print(vision[i, j], end=" ")  
        print()
    print(dir)
    return vision

def debug_state(snake_env):
    """return state of the game"""
    if len(snake_env.snake) < 3:
        print("Snake too short!")
        return None
        
    head = snake_env.snake[0]
    corps1 = snake_env.snake[1]
    corps2 = snake_env.snake[2]
    
    if not snake_env.green_apples or len(snake_env.green_apples) < 2:
        print("Not enough green apples!")
        return None
        
    apple1 = snake_env.green_apples[0]
    apple2 = snake_env.green_apples[1]
    red = snake_env.red_apples
    
    print("Head: ", head)
    print("Corps: ", corps1)
    print("Corps2: ", corps2)
    print("Apple1: ", apple1)
    print("Apple2: ", apple2)
    print("red apple: ", red)


if __name__ == "__main__":
    pg.init()
    snakeEnv = SnakeEnv(10)
    screen = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pg.display.set_caption("Learn2Slither - Snake RL")
    clock = pg.time.Clock()
    running = True
    paused = False
    
    while running:
        running, paused = key_event(running, paused, snakeEnv)
        
        if not paused:
            game_status = snakeEnv.update_snake()
            snakeEnv.score = len(snakeEnv.snake) - 3
            if not game_status:
                # print("Resetting game...")
                snakeEnv.reset()

        screen.fill(BLACK)

        # Dessiner la grille
        for i in range(0, WINDOW_SIZE, CELL_SIZE):
            for y in range(0, WINDOW_SIZE, CELL_SIZE):
                rect = pg.Rect(i, y, CELL_SIZE, CELL_SIZE)
                pg.draw.rect(screen, WHITE, rect, 1)
        
        # Dessiner les pommes vertes        
        for apple in snakeEnv.green_apples:
            pg.draw.rect(screen, GREEN, (apple[1] * CELL_SIZE, apple[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Dessiner la pomme rouge
        pg.draw.rect(screen, RED, (snakeEnv.red_apples[1] * CELL_SIZE, snakeEnv.red_apples[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Dessiner le serpent
        for seg in snakeEnv.snake:
            pg.draw.rect(screen, BLUE, (seg[1] * CELL_SIZE, seg[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Afficher l'état actuel
        # debug_state(snakeEnv)
        get_state(snakeEnv)
        print(snakeEnv.score)
        
        pg.display.flip()
        clock.tick(5)
    
    pg.quit()
