import torch
from tqdm import trange
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse

print("DEBUG: tournament.py script started")

from hex_engine import hexPosition
from networks import ActorCritic, ResNet, MiniResNet
from a2c_agent import get_agent_move as get_a2c_move
from mcts_agent import MCTS
from baseline_agents import RandomAgent, GreedyAgent, DefensiveAgent, AggressiveAgent

def load_player_func(args, agent_type, network_type=None, model_path=None, device=None, player_num=1):
    """Loads a model or baseline agent and returns a callable function."""
    print(f"DEBUG: Loading player {player_num} -> Type: {agent_type}, Network: {network_type}, Path: {model_path}")
    
    # Handle baseline agents
    if agent_type == 'greedy':
        greedy_agent = GreedyAgent()
        return lambda board, action_set, player: greedy_agent.select_move(board, action_set, player)
    if agent_type == 'random':
        random_agent = RandomAgent()
        return lambda board, action_set, player: random_agent.select_move(board, action_set, player)
    if agent_type == 'aggressive':
        aggressive_agent = AggressiveAgent()
        return lambda board, action_set, player: aggressive_agent.select_move(board, action_set, player)
    if agent_type == 'defensive':
        defensive_agent = DefensiveAgent()
        return lambda board, action_set, player: defensive_agent.select_move(board, action_set, player)

    # Handle neural network agents
    print(f"Loading Player {player_num} ({agent_type.upper()}): Network={network_type.upper()}, Model={model_path}")
    
    if network_type == 'cnn':
        agent = ActorCritic(args.board_size, args.board_size**2).to(device)
    elif network_type == 'resnet':
        agent = ResNet(args.board_size, args.board_size**2).to(device)
    elif network_type == 'miniresnet':
        agent = MiniResNet(args.board_size, args.board_size**2).to(device)
    else:
        raise ValueError(f"Unknown network type for {agent_type.upper()}: {network_type}")

    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
    except FileNotFoundError:
        print(f"FATAL: Model file not found at {model_path}.")
        exit()
    except Exception as e:
        print(f"FATAL: Error loading model {model_path}. Error: {e}")
        exit()

    if agent_type == 'a2c':
        return lambda board, action_set, player: get_a2c_move(agent, board, player, device, args.board_size)
    elif agent_type == 'mcts':
        mcts = MCTS(agent, device=device)
        def mcts_player_func(board, action_set, player):
            temp_env = hexPosition(args.board_size)
            temp_env.board = board
            temp_env.player = player
            action_probs = mcts.get_action_probs(temp_env, args.mcts_simulations, temp=0)
            return max(action_probs, key=action_probs.get)
        return mcts_player_func
    else:
        raise ValueError(f"Unknown agent type for {agent_type.upper()}: {agent_type}")


def run_tournament(args):
    """The main entry point for running a tournament between two agents."""
    print("DEBUG: Starting Tournament...")
    device = torch.device("mps" if torch.backends.mps.is_available() and args.environment == 'apple' else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Board Size: {args.board_size}x{args.board_size}\nDevice: {device}\n---------------------")

    # Load the primary player (P1)
    p1_func = load_player_func(args, args.p1_agent_type, args.p1_network_type, args.p1_model_path, device, player_num=1)
    print(f"\nLoaded Player 1: {args.p1_agent_type.upper()} from {args.p1_model_path}")
    
    # Determine opponents
    if args.p2_agent_type:
        opponents_to_test = [{'type': args.p2_agent_type, 'net': args.p2_network_type, 'path': args.p2_model_path}]
        match_title = f"{args.p1_agent_type.upper()} ({args.p1_network_type.upper()}) vs {args.p2_agent_type.upper()} ({args.p2_network_type.upper() if args.p2_network_type else ''})"
    else:
        opponents_to_test = [
            {'type': 'random', 'net': None, 'path': None},
            {'type': 'greedy', 'net': None, 'path': None},
            {'type': 'aggressive', 'net': None, 'path': None},
            {'type': 'defensive', 'net': None, 'path': None}
        ]
        match_title = f"{args.p1_agent_type.upper()} ({args.p1_network_type.upper()}) vs Baselines"

    all_results = []

    for opponent in opponents_to_test:
        opponent_agent_type = opponent['type']
        print(f"\n--- Starting Match vs {opponent_agent_type.upper()} ---")
        p2_func = load_player_func(args, opponent_agent_type, opponent['net'], opponent['path'], device, player_num=2)

        env = hexPosition(args.board_size)
        p1_as_white_wins = 0
        desc_p1_white = f"P1 ({args.p1_agent_type.upper()}) as White vs {opponent_agent_type.upper()}"
        for _ in trange(args.test_episodes // 2, desc=desc_p1_white):
            winner = env.machine_vs_machine_silent(machine1=p1_func, machine2=p2_func)
            if winner == 1: p1_as_white_wins += 1
        
        p1_as_black_wins = 0
        desc_p1_black = f"P1 ({args.p1_agent_type.upper()}) as Black vs {opponent_agent_type.upper()}"
        for _ in trange(args.test_episodes // 2, desc=desc_p1_black):
            winner = env.machine_vs_machine_silent(machine1=p2_func, machine2=p1_func)
            if winner == -1: p1_as_black_wins += 1

        total_wins = p1_as_white_wins + p1_as_black_wins
        win_rate = total_wins / args.test_episodes
        
        print(f"\n--- Match Stats vs {opponent_agent_type.upper()} ---")
        print(f"P1 Wins as White: {p1_as_white_wins}/{args.test_episodes//2}")
        print(f"P1 Wins as Black: {p1_as_black_wins}/{args.test_episodes//2}")
        
        all_results.append({
            'opponent': opponent_agent_type.upper(),
            'total_wins': total_wins,
            'win_rate': win_rate
        })
        print(f"Result vs {opponent_agent_type.upper()}: {total_wins}/{args.test_episodes} ({win_rate:.2%})")

    # --- Save and Plot Final Results ---
    results_filename_base = f"tournament_results_{args.p1_agent_type}_{args.p1_network_type}_vs_{args.p2_agent_type or 'baselines'}"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{results_filename_base}.csv', index=False)
    print(f"\nFull tournament results saved to {results_filename_base}.csv")
    
    # Generate and save the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['opponent'], results_df['win_rate'] * 100, color=['#66b3ff', '#99ff99', '#ffcc99', '#ff9999'])
    plt.ylabel('Win Rate (%)')
    plt.xlabel('Opponent')
    plt.title(f"{match_title} ({args.test_episodes} games per opponent)")
    plt.ylim(0, 100)
    for index, value in enumerate(results_df['win_rate'] * 100):
        plt.text(index, value + 1, f"{value:.1f}%", ha='center')
    
    plt.savefig(f'{results_filename_base}.png')
    print(f"Tournament results plot saved to {results_filename_base}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a tournament between two Hex agents.")
    parser.add_argument("--board_size", type=int, default=7, help="Size of the hex board")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of games to play")
    parser.add_argument("--environment", type=str, default="default", help="Environment for device selection ('apple' for mps)")

    # Player 1 arguments
    parser.add_argument("--p1_agent_type", type=str, required=True, choices=['a2c', 'mcts', 'greedy', 'random', 'aggressive', 'defensive'], help="Type of agent for Player 1")
    parser.add_argument("--p1_network_type", type=str, choices=['cnn', 'resnet', 'miniresnet'], help="Network type for Player 1 (if applicable)")
    parser.add_argument("--p1_model_path", type=str, help="Path to model for Player 1 (if applicable)")

    # Player 2 arguments
    parser.add_argument("--p2_agent_type", type=str, choices=['a2c', 'mcts', 'greedy', 'random', 'aggressive', 'defensive'], help="Type of agent for Player 2 (optional, defaults to all baselines)")
    parser.add_argument("--p2_network_type", type=str, choices=['cnn', 'resnet', 'miniresnet'], help="Network type for Player 2 (if applicable)")
    parser.add_argument("--p2_model_path", type=str, help="Path to model for Player 2 (if applicable)")
    
    # MCTS specific arguments
    parser.add_argument("--mcts_simulations", type=int, default=100, help="Number of simulations for MCTS agent")

    args = parser.parse_args()
    run_tournament(args) 