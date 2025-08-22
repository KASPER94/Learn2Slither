import torch
import random
import numpy as np
from collections import deque
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn2slither.env import SnakeEnv, Coord
from .model import Linear_QNet, QTrainer
from .helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9   #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() is exceed
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = (head[0], head[1] - 1)
        point_r = (head[0], head[1] + 1)
        point_u = (head[0] - 1, head[1])
        point_d = (head[0] + 1, head[1])
        
        dir_l = game.dir == "LEFT"
        dir_r = game.dir == "RIGHT"
        dir_u = game.dir == "UP"
        dir_d = game.dir == "DOWN"
        # 4 directions to move: left, right, up, down

        state = [
            #11 values
            # Danger straight
            (dir_r and game._collision(point_r)) or game._has_red_apple(point_r) or
            (dir_l and game._collision(point_l)) or game._has_red_apple(point_l) or
            (dir_u and game._collision(point_u)) or game._has_red_apple(point_u) or
            (dir_d and game._collision(point_d)) or game._has_red_apple(point_d),

            # Danger right
            (dir_u and game._collision(point_r)) or game._has_red_apple(point_r) or
            (dir_d and game._collision(point_l)) or game._has_red_apple(point_l) or
            (dir_l and game._collision(point_u)) or game._has_red_apple(point_u) or
            (dir_r and game._collision(point_d)) or game._has_red_apple(point_d),
            # Danger left
            (dir_u and game._collision(point_l)) or game._has_red_apple(point_l) or
            (dir_d and game._collision(point_r)) or game._has_red_apple(point_r) or
            (dir_r and game._collision(point_u)) or game._has_red_apple(point_u) or
            (dir_l and game._collision(point_d)) or game._has_red_apple(point_d),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            (game.green_apples[0][0] < game.snake[0][0]) if game.green_apples else False, # food left
            (game.green_apples[0][0] > game.snake[0][0]) if game.green_apples else False, # food right
            (game.green_apples[0][1] < game.snake[0][1]) if game.green_apples else False, # food up
            (game.green_apples[0][1] > game.snake[0][1]) if game.green_apples else False # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # one step only
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploitation /exploitation
        # self.epsilon = 80 - self.n_games
        self.epsilon = max(0.01, 0.9 * (0.995 ** self.n_games))
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


    def train_step(self, state, action, reward, next_state, done):
        pass

    @staticmethod
    def load(model_path: str) -> Linear_QNet:
        agent = DQNAgent()
        agent.model.load(model_path)
        return agent.model

    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent()
    game = SnakeEnv()
    while True:
        if agent.n_games > 600:
            break
        #get old state
        old_state = agent.get_state(game)

        #get move
        final_move = agent.get_action(old_state)

        #perform move and get new state
        reward, done, score = game.step_dqn(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, state_new, done)

        #remember all of this actions
        agent.remember(old_state, final_move, reward, state_new, done)

        if done:
            #train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                #agent.model.save()
            
            print(f"Game {agent.n_games}, Record: {record}, Score: {score}")
            
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score/agent.n_games)
            plot(plot_scores, plot_mean_scores)
    
    agent.model.save("model.pth")

if __name__ == "__main__":
    train()
    