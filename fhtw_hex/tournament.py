import torch
from tqdm import trange
import pandas as pd
import os
import matplotlib.pyplot as plt

from hex_engine import hexPosition
from networks import ActorCritic, ResNet
from a2c_agent import get_agent_move as get_a2c_move
from mcts_agent import MCTS
from submission.greedy_agent_adapter import create_greedy_player
from baseline_agents import RandomAgent

def load_player_func(args, player_num, device):
    """Loads a model and returns a function that takes a board and returns a move."""
    agent_type = getattr(args, f'p{player_num}_agent_type')
    
    # Handle baseline agents first
    if agent_type == 'greedy':
        print(f"Loading Player {player_num}: Agent=GREEDY")
        greedy_player_func = create_greedy_player()
        def greedy_wrapper(board, action_set, player):
            return greedy_player_func(board, action_set)
        return greedy_wrapper
    
    if agent_type == 'random':
        print(f"Loading Player {player_num}: Agent=RANDOM")
        random_agent = RandomAgent()
        def random_wrapper(board, action_set, player):
            return random_agent.select_move(board, action_set, player)
        return random_wrapper

    network_type = getattr(args, f'p{player_num}_network_type')
    model_path = getattr(args, f'p{player_num}_model_path')

    print(f"Loading Player {player_num}: Agent={agent_type.upper()}, Network={network_type.upper()}, Model={model_path}")

    # Instantiate the correct network
    if network_type == 'cnn':
        agent = ActorCritic(args.board_size, args.board_size**2).to(device)
    elif network_type == 'resnet':
        agent = ResNet(args.board_size, args.board_size**2).to(device)
    else:
        raise ValueError(f"Unknown network type for P{player_num}: {network_type}")

    # Load the saved weights
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
    except FileNotFoundError:
        print(f"FATAL: Model file not found at {model_path}. Cannot run tournament.")
        exit()
    except RuntimeError:
        print(f"WARNING: Mismatch loading {model_path} into {network_type.upper()} network. Attempting to load into other network type.")
        other_network_type = 'resnet' if network_type == 'cnn' else 'cnn'
        
        if other_network_type == 'cnn':
            agent = ActorCritic(args.board_size, args.board_size**2).to(device)
        else: # resnet
            agent = ResNet(args.board_size, args.board_size**2).to(device)
            
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()
            print(f"SUCCESS: Loaded {model_path} into fallback {other_network_type.upper()} network.")
        except Exception as e:
            print(f"FATAL: Failed to load model {model_path} into either network architecture. Error: {e}")
            exit()

    # Return a callable function that represents the player
    if agent_type == 'a2c':
        def a2c_player_func(board, action_set, player):
            return get_a2c_move(agent, board, player, device, args.board_size)
        return a2c_player_func

    elif agent_type == 'mcts':
        mcts = MCTS(agent, device=device)
        def mcts_player_func(board, action_set, player):
            temp_env = hexPosition(args.board_size)
            temp_env.board = board
            temp_env.player = player
            # Use a slightly higher number of simulations for tournament play for stronger performance
            action_probs = mcts.get_action_probs(temp_env, args.mcts_simulations * 2, temp=0)
            return max(action_probs, key=action_probs.get)
        return mcts_player_func
    
    else:
        raise ValueError(f"Unknown agent type for P{player_num}: {agent_type}")


def run_tournament(args):
    """The main entry point for running a tournament between two agents."""
    
    # Set device
    if args.environment == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.environment == 'gpu' and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Device: {device}")
    print("---------------------")
    
    # Load player functions
    player1_func = load_player_func(args, 1, device)
    player2_func = load_player_func(args, 2, device)

    print("\nAll players loaded successfully. Let the tournament begin!")

    env = hexPosition(args.board_size)
    tournament_log = []
    
    # --- Round 1: P1 is White, P2 is Black ---
    p1_label = f"P1 ({args.p1_agent_type.upper()})"
    p2_label = f"P2 ({args.p2_agent_type.upper()})"
    print(f"\n--- Round 1: {p1_label} is White, {p2_label} is Black ---")
    p1_as_white_wins = 0
    for _ in trange(args.test_episodes // 2, desc=f"{p1_label} as White"):
        winner = env.machine_vs_machine_silent(machine1=player1_func, machine2=player2_func)
        if winner == 1:
            p1_as_white_wins += 1
        tournament_log.append({'round': 1, 'p1_agent': args.p1_agent_type, 'p2_agent': args.p2_agent_type, 'winner': 'p1' if winner == 1 else 'p2'})
    
    # --- Round 2: P2 is White, P1 is Black ---
    print(f"\n--- Round 2: {p2_label} is White, {p1_label} is Black ---")
    p1_as_black_wins = 0
    for _ in trange(args.test_episodes // 2, desc=f"{p1_label} as Black"):
        winner = env.machine_vs_machine_silent(machine1=player2_func, machine2=player1_func)
        if winner == -1:
            p1_as_black_wins += 1
        tournament_log.append({'round': 2, 'p1_agent': args.p2_agent_type, 'p2_agent': args.p1_agent_type, 'winner': 'p1' if winner == -1 else 'p2'})

    # --- Save Detailed Log ---
    log_df = pd.DataFrame(tournament_log)
    log_df.to_csv('tournament_results.csv', index=False)
    print("\nDetailed tournament results saved to tournament_results.csv")

    # --- Final Results ---
    total_p1_wins = p1_as_white_wins + p1_as_black_wins
    total_games = args.test_episodes
    p1_win_rate = total_p1_wins / total_games if total_games > 0 else 0
    
    print("\n--- Tournament Results ---")
    print(f"Player 1 ({args.p1_agent_type.upper()}/{args.p1_network_type.upper()}) vs. Player 2 ({args.p2_agent_type.upper()}/{args.p2_network_type.upper()})")
    print(f"Total Games: {total_games}\n")
    print(f"P1 Wins as White: {p1_as_white_wins}/{args.test_episodes // 2}")
    print(f"P1 Wins as Black: {p1_as_black_wins}/{args.test_episodes // 2}")
    print("--------------------------")
    print(f"Overall P1 Win Rate: {total_p1_wins}/{total_games} = {p1_win_rate:.2%}")
    print("--------------------------\n")
    
    # --- Generate Pie Chart ---
    p1_name = args.p1_agent_type.upper()
    p2_name = args.p2_agent_type.upper()
    labels = [
        f'{p1_name} Wins as White ({p1_as_white_wins})', 
        f'{p1_name} Wins as Black ({p1_as_black_wins})',
        f'{p2_name} Wins as White ({args.test_episodes // 2 - p1_as_black_wins})',
        f'{p2_name} Wins as Black ({args.test_episodes // 2 - p1_as_white_wins})'
    ]
    sizes = [p1_as_white_wins, p1_as_black_wins, (args.test_episodes // 2 - p1_as_black_wins), (args.test_episodes // 2 - p1_as_white_wins)]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Tournament Results: {p1_name} (P1) vs {p2_name} (P2)')
    
    plt.savefig('tournament_results.png')
    print("Tournament results plot saved to tournament_results.png") 