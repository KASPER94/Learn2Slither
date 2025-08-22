import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
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
