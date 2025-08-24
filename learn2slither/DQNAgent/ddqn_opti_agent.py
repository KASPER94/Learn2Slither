import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#env
import sys
import os   
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn2slither.env import SnakeEnv

#plot settings
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

#Replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

#model
class Linear_QNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(n_features, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))  # Ajout de ReLU manquant
        x = self.linear3(x)  # Couche de sortie manquante !
        return x

#hyperparameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

n_actions = 3
n_features = 11
policy_net = Linear_QNet(n_features, n_actions).to(device)
target_net = Linear_QNet(n_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def get_random_action_vector():
    action = random.randint(0, 2)
    action_vector = [0, 0, 0]
    action_vector[action] = 1
    return action_vector

def select_action(state, env):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Obtenir les Q-values et l'action optimale
            q_values = policy_net(state)
            action_index = q_values.max(1)[1]  # Indice de l'action avec la plus grande Q-value
            
            # S'assurer que l'action est dans [0, 1, 2]
            action_index = torch.clamp(action_index, 0, 2)
            
            return action_index.view(1, 1)
    else:
        # Retourner un index d'action aléatoire (0, 1, ou 2)
        return torch.tensor([[random.randint(0, 2)]], device=device, dtype=torch.long)

#plotting - métriques utiles pour Snake
episode_scores = []
episode_rewards = []
episode_durations = []

# Figure globale pour éviter les multiples fenêtres
fig = None

def plot_training_metrics(show_result=False):
    global fig
    
    # Créer la figure une seule fois
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
    
    # Effacer le contenu précédent
    fig.clear()
    
    # Subplot 1: Scores (longueur du snake)
    ax1 = fig.add_subplot(1, 3, 1)
    if episode_scores:
        scores_t = torch.tensor(episode_scores, dtype=torch.float)
        ax1.plot(scores_t.numpy(), label='Score par épisode', color='blue')
        if len(scores_t) >= 10:
            means = scores_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            ax1.plot(means.numpy(), label='Moyenne mobile (10 épisodes)', alpha=0.7, color='red')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Score (longueur snake)')
        ax1.set_title('Performance: Score du Snake')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Récompenses cumulées
    ax2 = fig.add_subplot(1, 3, 2)
    if episode_rewards:
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        ax2.plot(rewards_t.numpy(), label='Récompense par épisode', color='green')
        if len(rewards_t) >= 10:
            means = rewards_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            ax2.plot(means.numpy(), label='Moyenne mobile (10 épisodes)', alpha=0.7, color='darkgreen')
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Récompense cumulée')
        ax2.set_title('Apprentissage: Récompenses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Durée de survie
    ax3 = fig.add_subplot(1, 3, 3)
    if episode_durations:
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        ax3.plot(durations_t.numpy(), label='Durée par épisode', color='orange')
        if len(durations_t) >= 10:
            means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            ax3.plot(means.numpy(), label='Moyenne mobile (10 épisodes)', alpha=0.7, color='darkorange')
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Steps de survie')
        ax3.set_title('Survie: Durée des épisodes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if is_ipython:
        if not show_result:
            display.display(fig)
            display.clear_output(wait=True)
        else:
            display.display(fig)
    else:
        plt.draw()
        plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


#training

def train(n_episodes=600):
    env = SnakeEnv()
    
    for i_episode in range(n_episodes):
        # Initialize the environment and get it's initial state
        env.reset()
        state = env._get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_reward = 0.0  # Récompense cumulée pour cet épisode

        for t in count():
            # Select and perform an action
            action = select_action(state, env)
            
            # Convertir l'action tensor en vecteur one-hot pour step_dqn
            action_index = action.item()
            action_vector = [0, 0, 0]
            action_vector[action_index] = 1
            
            reward, done, score = env.step_dqn(action_vector)
            episode_reward += reward  # Accumuler les récompenses
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = env._get_state()
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward, done)
            state = next_state
            optimize_model()
            
            # Soft update du target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # Collecter les métriques de fin d'épisode
                final_score = len(env.snake)  # Score = longueur du snake
                episode_scores.append(final_score)
                episode_rewards.append(episode_reward)
                episode_durations.append(t + 1)
                
                # Affichage des stats tous les 10 épisodes
                if (i_episode + 1) % 10 == 0:
                    avg_score = sum(episode_scores[-10:]) / min(10, len(episode_scores))
                    avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                    print(f"Épisode {i_episode + 1}: Score={final_score}, "
                          f"Récompense={episode_reward:.1f}, "
                          f"Moy. Score={avg_score:.1f}, Moy. Récompense={avg_reward:.1f}")
                
                # Mettre à jour les graphiques tous les 25 épisodes
                if (i_episode + 1) % 25 == 0:
                    plot_training_metrics()
                
                break
    
    # Entraînement terminé
    print('Entraînement terminé!')
    plot_training_metrics(show_result=True)
    plt.ioff()
    plt.show()
    
    # Sauvegarder le modèle
    torch.save(policy_net.state_dict(), 'models/ddqn_opti_snake_net.pth')
    print("Modèle sauvegardé: models/ddqn_opti_snake_net.pth")


if __name__ == "__main__":
    train()






            

