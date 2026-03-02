# play.py
import time
import torch
from agent import Agent
from game import SnakeGame, SPEED


def play(speed_ms: int = SPEED) -> None:
    """
    Joue au Snake avec le modèle entraîné à une vitesse visible en continu.
    Appuyez sur Ctrl+C pour arrêter.
    
    Args:
        speed_ms: Délai entre chaque frame en millisecondes
    """
    agent = Agent()
    game = SnakeGame()
    
    # Charger le modèle entraîné
    try:
        agent.model.load_state_dict(torch.load('model.pth'))
        agent.model.eval()
        print("Modèle chargé depuis model.pth")
        print(f"Vitesse: {speed_ms}ms par frame")
        print("Mode: Jeu continu (Ctrl+C pour arrêter)\n")
    except FileNotFoundError:
        print("Erreur: model.pth introuvable. Entraînez d'abord le modèle avec train.py")
        return
    
    total_score = 0
    max_score = 0
    game_num = 0
    
    try:
        while game.is_open():
            game_num += 1
            game.reset()
            game_over = False
            score = 0
            
            print(f"Partie {game_num} en cours...")
            
            while not game_over and game.is_open():
                # Obtenir l'état actuel
                state = agent.get_state(game)
                
                # Obtenir l'action du modèle (sans exploration aléatoire)
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float).to(agent.model.device)
                    prediction = agent.model(state_tensor)
                    move = int(torch.argmax(prediction).item())
                
                final_move = [0, 0, 0]
                final_move[move] = 1
                
                # Jouer l'action
                reward, game_over, score = game.play_step(final_move)
                if not game.is_open():
                    break
                
                # Délai pour rendre le jeu visible
                time.sleep(speed_ms / 1000.0)
            
            print(f"Partie {game_num} terminée - Score: {score}")
            total_score += score
            max_score = max(max_score, score)
            print(f"Score moyen: {total_score / game_num:.2f} | Meilleur: {max_score}\n")

        if game_num > 0:
            print("Fenêtre fermée, arrêt du mode continu.")
    
    except KeyboardInterrupt:
        print(f"\n\n--- Résultats Finaux ---")
        print(f"Nombre de parties: {game_num}")
        if game_num > 0:
            print(f"Score moyen: {total_score / game_num:.2f}")
        else:
            print("Score moyen: 0.00")
        print(f"Meilleur score: {max_score}")
        print(f"Score total: {total_score}")


if __name__ == "__main__":
    # Vous pouvez ajuster la vitesse
    play(speed_ms=50)
