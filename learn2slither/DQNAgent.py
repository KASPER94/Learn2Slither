import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.stack_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x):
        return self.stack_layers(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # data from DQN best practices: replay memory
        self.memory = deque(maxlen=memory_size)
        
        # data from DQN best practices: main and target networks
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # data from DQN best practices: update target network
        self.update_target_model()
    
    def update_target_model(self):
        """Met à jour le réseau cible avec les poids du réseau principal."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire de replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Choisit une action selon la politique ε-greedy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # data from PyTorch: convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Entraîne le modèle sur un batch d'expériences."""
        if len(self.memory) < self.batch_size:
            return
        
        # data from DQN: sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        # data from DQN: compute current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # data from DQN: compute target Q values
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # data from PyTorch: compute loss and update
        loss = nn.MSELoss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # data from DQN: decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def end_episode(self):
        """Appelé à la fin de chaque épisode."""
        # data from DQN: update target network periodically
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
        else:
            self.episode_count = 1
        
        # Update target network every 10 episodes
        if self.episode_count % 10 == 0:
            self.update_target_model()
    
    def save(self, path):
        """Sauvegarde le modèle principal."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Charge le modèle principal."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()