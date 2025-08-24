#!/usr/bin/env python3
"""
Script d'inférence pour tester le modèle DDQN entraîné
Affiche le jeu avec pygame et le score en temps réel
"""

import pygame as pg
import sys
import os
import time

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from learn2slither.env import SnakeEnv
from learn2slither.DQNAgent.ddqn_agent import DDQNAgent
from learn2slither.render import draw_scene
from learn2slither.config import CELL_SIZE, COLORS, GRID_SIZE


class DDQNInference:
    """
    Classe pour tester l'inférence du modèle DDQN avec interface pygame
    """
    
    def __init__(self, model_path: str = "./models/ddqn_model.pth"):
        """
        Initialise l'inférence avec le modèle pré-entraîné
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        self.model_path = model_path
        self.agent = None
        self.env = SnakeEnv()
        
        # Configuration pygame
        pg.init()
        self.window_size = GRID_SIZE * CELL_SIZE
        self.screen = pg.display.set_mode((self.window_size, self.window_size + 80))  # +100 pour le score
        pg.display.set_caption("DDQN Snake - Inférence")
        self.clock = pg.time.Clock()
        self.font = pg.font.Font(None, 36)
        
        # Statistiques
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        
    def load_model(self):
        """
        Charge le modèle DDQN pré-entraîné
        """
        try:
            if os.path.exists(self.model_path):
                self.agent = DDQNAgent.load(self.model_path)
                print(f"Modèle chargé depuis: {self.model_path}")
                return True
            else:
                print(f"Erreur: Modèle non trouvé à {self.model_path}")
                print("Veuillez d'abord entraîner un modèle avec ddqn_agent.py")
                return False
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def draw_score(self):
        """
        Affiche le score et les statistiques en haut à gauche
        """
        current_score = len(self.env.snake)
        
        # Score actuel
        score_text = self.font.render(f"Score: {current_score}", True, COLORS["WHITE"])
        self.screen.blit(score_text, (10, self.window_size + 10))
        
        # Meilleur score
        best_text = self.font.render(f"Meilleur: {self.best_score}", True, COLORS["GREEN"])
        self.screen.blit(best_text, (10, self.window_size + 40))
        
        # Nombre de parties
        games_text = self.font.render(f"Parties: {self.games_played}", True, COLORS["BLUE"])
        self.screen.blit(games_text, (200, self.window_size + 10))
        
        # Score moyen
        avg_score = self.total_score / max(1, self.games_played)
        avg_text = self.font.render(f"Moyenne: {avg_score:.1f}", True, COLORS["BLUE"])
        self.screen.blit(avg_text, (200, self.window_size + 40))
    
    def run_inference(self, max_games: int = 10, fps: int = 10):
        """
        Lance l'inférence avec affichage pygame
        
        Args:
            max_games: Nombre maximum de parties à jouer
            fps: Images par seconde (vitesse du jeu)
        """
        if not self.load_model():
            return
        
        print(f"Démarrage de l'inférence DDQN - {max_games} parties max")
        print("Appuyez sur ESPACE pour accélérer, ESC pour quitter")
        
        running = True
        game_active = True
        fast_mode = False
        
        while running and self.games_played < max_games:
            # Gestion des événements
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_SPACE:
                        fast_mode = not fast_mode
                        print(f"Mode rapide: {'ON' if fast_mode else 'OFF'}")
            
            if game_active:
                # Obtenir l'état actuel
                state = self.agent.get_state(self.env)
                
                # L'agent choisit une action
                action = self.agent.get_action(state)
                
                # Exécuter l'action
                reward, done, score = self.env.step_dqn(action)
                
                # Affichage
                draw_scene(self.screen, self.env)
                self.draw_score()
                pg.display.flip()
                
                if done:
                    # Fin de partie
                    final_score = len(self.env.snake)
                    self.games_played += 1
                    self.total_score += final_score
                    
                    if final_score > self.best_score:
                        self.best_score = final_score
                        print(f"🎉 Nouveau record: {final_score}!")
                    
                    print(f"Partie {self.games_played}: Score = {final_score}")
                    
                    # Pause entre les parties
                    time.sleep(1.0)
                    self.env.reset()
                    game_active = True
            
            # Contrôle de la vitesse
            if fast_mode:
                self.clock.tick(fps * 5)  # 5x plus rapide
            else:
                self.clock.tick(fps)
        
        # Statistiques finales
        self.show_final_stats()
        pg.quit()
    
    def show_final_stats(self):
        """
        Affiche les statistiques finales
        """
        print("\n" + "="*50)
        print("STATISTIQUES FINALES")
        print("="*50)
        print(f"Parties jouées: {self.games_played}")
        print(f"Score total: {self.total_score}")
        print(f"Score moyen: {self.total_score / max(1, self.games_played):.2f}")
        print(f"Meilleur score: {self.best_score}")
        print("="*50)


def main():
    """
    Point d'entrée principal
    """
    # Vérifier si un modèle existe
    model_path = "./models/ddqn_model.pth"
    
    if not os.path.exists(model_path):
        print("Aucun modèle DDQN trouvé.")
        print("Voulez-vous d'abord entraîner un modèle ? (y/n)")
        response = input().lower()
        
        if response == 'y':
            print("Lancement de l'entraînement...")
            from learn2slither.DQNAgent.ddqn_agent import train
            train()
        else:
            print("Arrêt du programme.")
            return
    
    # Lancer l'inférence
    inference = DDQNInference(model_path)
    
    try:
        # Demander les paramètres à l'utilisateur
        print("Configuration de l'inférence:")
        max_games = int(input("Nombre de parties (défaut: 10): ") or "10")
        fps = int(input("Vitesse FPS (défaut: 10): ") or "10")
        
        inference.run_inference(max_games=max_games, fps=fps)
        
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        pg.quit()


if __name__ == "__main__":
    main() 