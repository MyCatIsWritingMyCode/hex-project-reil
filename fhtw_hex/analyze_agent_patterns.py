import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tqdm import trange

from hex_engine import hexPosition
from networks import ActorCritic
from a2c_agent import get_agent_move
from submission.greedy_agent_adapter import create_greedy_player
from plotting import plot_board_state

# --- Agent Definitions ---

AGENTS = {
    "a2c_v1_static": lambda device, board_size: (
        lambda board, action_set: get_agent_move(
            load_a2c_agent("a2c_v1_static_agent.pth", device, board_size),
            board, get_player_from_board(board), device, board_size
        )
    ),
    "a2c_v2_dynamic": lambda device, board_size: (
        lambda board, action_set: get_agent_move(
            load_a2c_agent("a2c_v2_agent.pth", device, board_size),
            board, get_player_from_board(board), device, board_size
        )
    ),
}

# --- Helper Functions ---

def load_a2c_agent(path, device, board_size):
    """Loads a saved A2C agent model."""
    try:
        agent = ActorCritic(board_size, board_size**2).to(device)
        agent.load_state_dict(torch.load(path, map_location=device))
        agent.eval()
        return agent
    except FileNotFoundError:
        print(f"Warning: Model file not found at {path}. Skipping agent.")
        return None

def get_player_from_board(board):
    """Determines the current player by counting stones."""
    num_white = sum(row.count(1) for row in board)
    num_black = sum(row.count(-1) for row in board)
    return 1 if num_white == num_black else -1

def play_and_record(agent1_func, agent2_func, board_size, n_games, save_snapshots=False, p1_name="", p2_name=""):
    """Plays N games between two agents and records move frequencies."""
    move_counts_p1 = np.zeros((board_size, board_size))
    move_counts_p2 = np.zeros((board_size, board_size))
    
    snapshot_dir = "board_snapshots"
    if save_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)

    for game_num in trange(n_games, desc="Simulating Games"):
        env = hexPosition(board_size)
        
        while env.winner == 0:
            player = env.player
            
            if player == 1:
                move = agent1_func(env.board, env.get_action_space())
            else:
                move = agent2_func(env.board, env.get_action_space())
            
            # Record the move for the correct player
            if player == 1:
                move_counts_p1[move] += 1
            else:
                move_counts_p2[move] += 1

            env.move(move)
        
        # Save a snapshot of the final board state for a few games
        if save_snapshots and game_num < 3:
            filename = f"{snapshot_dir}/{p1_name}(P1)_vs_{p2_name}(P2)_game{game_num+1}_winner{env.winner}.png"
            plot_board_state(env.board, filename)
            
    return move_counts_p1, move_counts_p2

def plot_heatmap(move_counts, title, filename):
    """Generates and saves a heatmap plot."""
    if np.sum(move_counts) == 0:
        print(f"Skipping heatmap for '{title}' due to no data.")
        return
        
    plt.figure(figsize=(8, 6))
    plt.imshow(move_counts, cmap='hot', interpolation='nearest')
    plt.title(title, fontsize=14)
    plt.colorbar(label='Move Frequency')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig(filename)
    plt.close()

# --- Main Execution ---

def main():
    """Main function to run all pairwise agent matchups."""
    board_size = 7
    n_games_per_matchup = 100
    save_board_snapshots = True
    device = torch.device('cpu')
    
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    agent_names = list(AGENTS.keys())
    
    for p1_name in agent_names:
        for p2_name in agent_names:
            print(f"\n--- Matchup: {p1_name} (as P1) vs. {p2_name} (as P2) ---")
            
            agent1_loader = AGENTS.get(p1_name)
            agent2_loader = AGENTS.get(p2_name)
            
            if not agent1_loader or not agent2_loader:
                print("Skipping matchup due to missing agent loader.")
                continue

            agent1_func = agent1_loader(device, board_size)
            agent2_func = agent2_loader(device, board_size)

            if agent1_func is None or agent2_func is None:
                print("Skipping matchup due to missing model file.")
                continue

            move_counts_p1, move_counts_p2 = play_and_record(
                agent1_func, agent2_func, board_size, n_games_per_matchup,
                save_snapshots=save_board_snapshots, p1_name=p1_name, p2_name=p2_name
            )
            
            # The heatmap always shows the move distribution for the agent in that role
            plot_heatmap(
                move_counts_p1, 
                f"Move Distribution for {p1_name} (as P1)", 
                f"{output_dir}/{p1_name}_as_P1_vs_{p2_name}.png"
            )
            plot_heatmap(
                move_counts_p2, 
                f"Move Distribution for {p2_name} (as P2)", 
                f"{output_dir}/{p2_name}_as_P2_vs_{p1_name}.png"
            )

    print("\n--- Analysis Complete ---")
    print(f"All heatmaps and snapshots saved in '{output_dir}/' and 'board_snapshots/'")

if __name__ == '__main__':
    main() 