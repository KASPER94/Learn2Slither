import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class TargetNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        file_name = os.path.join(file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()
        return self
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + y(gamma) * max(next_predicted Q value)
        # pred.clone()
        # preds[argmax(action)] = Q_new
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class DDQNTrainer:
    """
    Trainer pour Double DQN qui utilise deux réseaux pour réduire le biais de surestimation
    - main_model: réseau principal utilisé pour sélectionner les actions
    - target_model: réseau target utilisé pour évaluer les valeurs Q
    """
    def __init__(self, main_model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.main_model = main_model
        self.target_model = target_model
        self.optimizer = optim.Adam(main_model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        """
        Étape d'entraînement Double DQN optimisée avec traitement vectorisé
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.bool)

        # Gestion des dimensions : ajouter batch dimension si nécessaire
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0) if reward.dim() == 0 else reward
            done = done.unsqueeze(0) if done.dim() == 0 else done

        # Conversion one-hot → indices si nécessaire
        action = torch.as_tensor(action)
        if action.dim() == 1 and len(action) > 1:  # One-hot vector
            action = action.argmax().unsqueeze(0)
        elif action.dim() == 2:  # Batch of one-hot vectors
            action = action.argmax(dim=1)
        else:  # Déjà des indices
            action = action.long()
            if action.dim() == 0:  # Scalaire
                action = action.unsqueeze(0)

        # Q(s,a) actuel (réseau principal)
        q_all = self.main_model(state)                           # [B, A]
        q_sa = q_all.gather(1, action.unsqueeze(1)).squeeze(1)   # [B]

        # Cible DDQN (sélection par main, évaluation par target), sans gradient
        with torch.no_grad():
            next_q_main = self.main_model(next_state)            # [B, A]
            best_next_act = next_q_main.argmax(dim=1)            # [B]
            next_q_target = self.target_model(next_state)        # [B, A]
            next_q_t_sa = next_q_target.gather(1, best_next_act.unsqueeze(1)).squeeze(1)
            next_q_t_sa = next_q_t_sa * (~done)                 # 0 si terminal
            target_q = reward + self.gamma * next_q_t_sa         # [B]

        # Perte et optimisation
        self.optimizer.zero_grad(set_to_none=True)
        loss = nn.SmoothL1Loss()(q_sa, target_q)                # Huber plus stable que MSE
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), 10.0)
        self.optimizer.step()

        # Soft update à chaque step
        self.soft_update()

    def soft_update(self, tau=0.005):
        """
        Mise à jour douce du réseau target à chaque étape d'entraînement
        """
        with torch.no_grad():
            for target_param, main_param in zip(self.target_model.parameters(), self.main_model.parameters()):
                target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
        
    def update_target_network(self):
        """
        Copie complète des poids (gardé pour compatibilité mais non utilisé)
        """
        self.target_model.load_state_dict(self.main_model.state_dict())


class Inference:
    def __init__(self, model, input_size, hidden_size, output_size):
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.model.load(model)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

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

        state = np.array(state)
        state_tensor = torch.tensor(state, dtype=torch.float)
        state_tensor = state_tensor.unsqueeze(0)
        return state_tensor
    
    def get_action(self, state):
        action_vector = [0, 0, 0]
        with torch.no_grad():
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            action_vector[action] = 1
            print(f"Action: {action}")
        return action_vector
