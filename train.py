from agent import Agent
from game import SnakeGame

DEFAULT_MAX_GAMES = 10000


def train(max_games: int = DEFAULT_MAX_GAMES) -> None:
    agent = Agent()
    game = SnakeGame()
    model_path = "models/model.pth"

    try:
        while agent.n_games < max_games:
            if not game.is_open():
                agent.model.save(model_path)
                print(f"Training stopped: game window closed. Model saved to {model_path}")
                return

            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)

            if not game.is_open():
                agent.model.save(model_path)
                print(f"Training stopped: game window closed. Model saved to {model_path}")
                return

            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                print("Game", agent.n_games, "Score", score)
    except KeyboardInterrupt:
        agent.model.save(model_path)
        print(f"Training interrupted. Model saved to {model_path}")
        return

    agent.model.save(model_path)
    print(f"Training finished ({agent.n_games}/{max_games} games). Model saved to {model_path}")


if __name__ == "__main__":
    train()