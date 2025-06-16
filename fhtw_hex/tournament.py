import torch
from tqdm import trange
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse

from hex_engine import hexPosition
from networks import ActorCritic, ResNet
from a2c_agent import get_agent_move as get_a2c_move
from mcts_agent import MCTS
from submission.greedy_agent_adapter import create_greedy_player
from baseline_agents import RandomAgent, GreedyAgent, DefensiveAgent, AggressiveAgent

def load_player_func(args, agent_type, network_type=None, model_path=None, device=None, player_num=1):
    """Loads a model or baseline agent and returns a callable function."""
    
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