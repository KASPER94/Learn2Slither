#!/usr/bin/env python3
"""
Script de test pour comparer les performances entre DQN et Double DQN
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from learn2slither.DQNAgent.dqn_agent import DQNAgent
from learn2slither.DQNAgent.ddqn_agent import DDQNAgent
from learn2slither.env import SnakeEnv
import matplotlib.pyplot as plt
import numpy as np


def test_agent(agent_class, agent_name, num_games=50):
    """
    Teste un agent sur un nombre donné de parties
    """
    print(f"\n=== Test de {agent_name} ===")
    
    # Créer l'agent et l'environnement
    if agent_class == DDQNAgent:
        agent = DDQNAgent()
    else:
        agent = DQNAgent()
    
    game = SnakeEnv()
    scores = []
    
    for game_num in range(num_games):
        game.reset()
        score = 0
        steps = 0
        max_steps = 1000  # Éviter les boucles infinies
        
        while not game.game_over and steps < max_steps:
            # Obtenir l'état et l'action
            state = agent.get_state(game)
            action = agent.get_action(state)
            
            # Exécuter l'action
            reward, done, current_score = game.step_dqn(action)
            score = current_score
            steps += 1
            
            if done:
                break
        
        scores.append(score)
        if (game_num + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Parties {game_num-8}-{game_num+1}: Score moyen = {avg_score:.2f}")
    
    avg_score = np.mean(scores)
    max_score = max(scores)
    min_score = min(scores)
    
    print(f"\n{agent_name} - Résultats sur {num_games} parties:")
    print(f"Score moyen: {avg_score:.2f}")
    print(f"Score maximum: {max_score}")
    print(f"Score minimum: {min_score}")
    print(f"Écart-type: {np.std(scores):.2f}")
    
    return scores, avg_score


def compare_agents():
    """
    Compare les performances entre DQN et Double DQN
    """
    print("Comparaison entre DQN et Double DQN")
    print("=" * 50)
    
    # Tester DQN classique
    dqn_scores, dqn_avg = test_agent(DQNAgent, "DQN Classique", num_games=30)
    
    # Tester Double DQN
    ddqn_scores, ddqn_avg = test_agent(DDQNAgent, "Double DQN", num_games=30)
    
    # Afficher la comparaison
    print(f"\n=== COMPARAISON ===")
    print(f"DQN Classique - Score moyen: {dqn_avg:.2f}")
    print(f"Double DQN - Score moyen: {ddqn_avg:.2f}")
    
    improvement = ((ddqn_avg - dqn_avg) / dqn_avg) * 100 if dqn_avg > 0 else 0
    print(f"Amélioration: {improvement:+.1f}%")
    
    # Créer un graphique de comparaison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dqn_scores, label='DQN Classique', alpha=0.7)
    plt.plot(ddqn_scores, label='Double DQN', alpha=0.7)
    plt.xlabel('Partie')
    plt.ylabel('Score')
    plt.title('Comparaison des scores par partie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([dqn_scores, ddqn_scores], labels=['DQN', 'Double DQN'])
    plt.ylabel('Score')
    plt.title('Distribution des scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_vs_ddqn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dqn_scores, ddqn_scores


def train_ddqn_demo(num_games=3000):
    from learn2slither.DQNAgent.ddqn_agent import train as train_ddqn
    train_ddqn()


if __name__ == "__main__":
    print("Script de test Double DQN")
    print("Choisissez une option:")
    print("1. Comparer DQN vs Double DQN (agents non-entraînés)")
    print("2. Démo d'entraînement Double DQN")
    print("3. Les deux")
    
    choice = input("Votre choix (1/2/3): ").strip()
    
    if choice in ["1", "3"]:
        compare_agents()
    
    if choice in ["2", "3"]:
        train_ddqn_demo()
    
    print("\nTest terminé!") 