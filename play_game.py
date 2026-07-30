"""
play_game.py
-------------
Interactive command-line game: YOU (human) play Tic-Tac-Toe against
the trained Q-learning agent. The agent plays as "X" and moves greedily
(always picking its best known move); you play as "O".

Usage:
    python src/play_game.py
"""

import os

from environment import TicTacToeEnv, AGENT, OPPONENT
from q_learning_agent import QLearningAgent

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_agent():
    model_path = os.path.join(MODELS_DIR, "q_table.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained Q-table found. Run train_agent.py first to create "
            "models/q_table.pkl"
        )
    agent = QLearningAgent()
    agent.load(model_path)
    return agent


def prompt_human_move(env):
    """Asks the human player to choose a valid empty cell (0-8)."""
    valid_actions = env.get_valid_actions()
    while True:
        try:
            move = int(input(f"Your move {valid_actions}: ").strip())
        except ValueError:
            print("  Please enter a number between 0 and 8.")
            continue
        if move in valid_actions:
            return move
        print(f"  Invalid move. Choose one of: {valid_actions}")


def print_board_positions():
    """Shows the player which number corresponds to which board cell."""
    print("\nBoard cell positions:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 \n")


def main():
    print("=" * 50)
    print(" TIC-TAC-TOE vs. Trained RL Agent")
    print(" You are 'O', the agent is 'X'. Agent moves first.")
    print("=" * 50)

    agent = load_agent()
    env = TicTacToeEnv()
    state = env.reset()
    print_board_positions()

    done = False
    while not done:
        # --- Agent's turn (goes first, greedy policy) ---
        valid_actions = env.get_valid_actions()
        agent_action = agent.choose_action(state, valid_actions, greedy=True)
        state, reward, done, info = env.step(agent_action, AGENT)

        print("\nAgent's move:")
        env.render()

        if done:
            break

        # --- Human's turn ---
        human_action = prompt_human_move(env)
        state, reward, done, info = env.step(human_action, OPPONENT)

        print("\nYour move:")
        env.render()

    # --- Announce result ---
    print("\n" + "-" * 50)
    if info.get("winner") == "agent":
        print("Result: The AGENT wins! Better luck next time.")
    elif info.get("winner") == "opponent":
        print("Result: YOU WIN! You beat the trained agent.")
    elif info.get("winner") == "draw":
        print("Result: It's a DRAW!")
    print("-" * 50)


if __name__ == "__main__":
    main()
