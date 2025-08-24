import torch
import random
import numpy as np
from collections import deque
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn2slither.env import SnakeEnv, Coord
from .model import Linear_QNet, DDQNTrainer
from .helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TARGET_UPDATE_FREQUENCY = 100  # Fréquence de mise à jour du réseau target


class DDQNAgent:
    """
    Agent Double DQN avec deux réseaux pour améliorer la stabilité d'entraînement
    """
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() if exceed
        
        # Deux réseaux: principal et target
        self.main_model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        
        # Initialiser le réseau target avec les mêmes poids que le réseau principal
        self.target_model.load_state_dict(self.main_model.state_dict())
        
        # Trainer Double DQN
        self.trainer = DDQNTrainer(self.main_model, self.target_model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Extrait l'état du jeu sous forme de vecteur de 11 valeurs
        """
        head = game.snake[0]
        point_l = (head[0], head[1] - 1)
        point_r = (head[0], head[1] + 1)
        point_u = (head[0] - 1, head[1])
        point_d = (head[0] + 1, head[1])
        
        dir_l = game.dir == "LEFT"
        dir_r = game.dir == "RIGHT"
        dir_u = game.dir == "UP"
        dir_d = game.dir == "DOWN"

        state = [
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
            (game.green_apples[0][0] < game.snake[0][0]) if game.green_apples else False,  # food left
            (game.green_apples[0][0] > game.snake[0][0]) if game.green_apples else False,  # food right
            (game.green_apples[0][1] < game.snake[0][1]) if game.green_apples else False,  # food up
            (game.green_apples[0][1] > game.snake[0][1]) if game.green_apples else False   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Stocke l'expérience dans la mémoire de replay
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Entraîne sur un batch d'expériences de la mémoire de replay
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Entraîne sur une seule expérience
        """
        self.trainer.train_step(state, action, reward, next_state, done)
    

    def get_action(self, state):
        """
        Sélectionne une action en utilisant epsilon-greedy avec le réseau principal
        """
        # Décroissance exponentielle de epsilon
        self.epsilon = max(0.01, 0.9 * (0.995 ** self.n_games))
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Action aléatoire (exploration)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Action basée sur le réseau principal (exploitation)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.main_model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def update_target_network(self):
        """
        Met à jour le réseau target avec les poids du réseau principal
        """
        self.trainer.update_target_network()

    @staticmethod
    def load(model_path: str):
        """
        Charge un modèle pré-entraîné
        """
        agent = DDQNAgent()
        agent.main_model.load(model_path)
        agent.target_model.load_state_dict(agent.main_model.state_dict())
        return agent


def train():
    """
    Fonction d'entraînement principale pour Double DQN
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DDQNAgent()
    game = SnakeEnv()
    
    while True:
        if agent.n_games > 600:
            break
            
        # Obtenir l'état actuel
        old_state = agent.get_state(game)

        # Obtenir l'action
        final_move = agent.get_action(old_state)

        # Exécuter l'action et obtenir le nouvel état
        reward, done, score = game.step_dqn(final_move)
        state_new = agent.get_state(game)

        # Entraînement sur l'expérience immédiate
        agent.train_short_memory(old_state, final_move, reward, state_new, done)

        # Stocker l'expérience
        agent.remember(old_state, final_move, reward, state_new, done)

        if done:
            # Fin de partie: entraînement sur batch et mise à jour
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # Mise à jour périodique du réseau target
            if agent.n_games % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
                print(f"Réseau target mis à jour au jeu {agent.n_games}")
            
            if score > record:
                record = score
            
            print(f"Jeu {agent.n_games}, Record: {record}, Score: {score}, Epsilon: {agent.epsilon:.3f}")
            
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot(plot_scores, plot_mean_scores)
    
    # Sauvegarder le modèle final
    agent.main_model.save("ddqn_model.pth")
    print("Modèle Double DQN sauvegardé sous 'ddqn_model.pth'")


if __name__ == "__main__":
    train()
        