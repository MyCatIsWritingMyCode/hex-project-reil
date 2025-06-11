import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Hex AI Engine', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Core arguments
    parser.add_argument('--agent-type', type=str, default='a2c', choices=['a2c', 'mcts'], help='The type of agent to use (for train/play/test modes).')
    parser.add_argument('--network', type=str, default='cnn', choices=['cnn', 'resnet'], help='The network architecture to use.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'test', 'tournament'], help='The mode to run the agent in.')
    parser.add_argument('--board-size', type=int, default=7, help='Size of the Hex board.')
    parser.add_argument('--model-path', type=str, help='Path for the primary model.')
    
    # Tournament specific arguments
    parser.add_argument('--p1-agent-type', type=str, default='a2c', choices=['a2c', 'mcts'], help='Agent type for Player 1 in tournament.')
    parser.add_argument('--p1-network-type', type=str, default='cnn', choices=['cnn', 'resnet'], help='Network for Player 1.')
    parser.add_argument('--p1-model-path', type=str, default='a2c_hex_agent.pth', help='Model path for Player 1.')
    parser.add_argument('--p2-agent-type', type=str, default='mcts', choices=['a2c', 'mcts'], help='Agent type for Player 2 in tournament.')
    parser.add_argument('--p2-network-type', type=str, default='resnet', choices=['cnn', 'resnet'], help='Network for Player 2.')
    parser.add_argument('--p2-model-path', type=str, default='mcts_hex_agent.pth', help='Model path for Player 2.')

    # Training arguments
    parser.add_argument('--n-episodes', type=int, default=1000, help='Number of episodes for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    
    # A2C specific arguments
    parser.add_argument('--a2c-gamma', type=float, default=0.99, help='Discount factor for A2C.')
    
    # MCTS specific arguments
    parser.add_argument('--mcts-simulations', type=int, default=50, help='Number of MCTS simulations per move.')
    parser.add_argument('--mcts-epochs', type=int, default=10, help='Number of training epochs per iteration for MCTS.')
    parser.add_argument('--mcts-batch-size', type=int, default=64, help='Batch size for MCTS training.')

    # Play/Test arguments
    parser.add_argument('--human-player', type=int, default=1, choices=[1, -1], help='Choose to be player 1 (white) or -1 (black) in play mode.')
    parser.add_argument('--test-episodes', type=int, default=100, help='Number of episodes for testing.')
    parser.add_argument('--environment', type=str, default='apple', choices=['windows', 'apple', 'kaggle'], help='The training environment to set the device.')

    args = parser.parse_args()

    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = f"{args.agent_type}_hex_agent.pth"

    print(f"--- Hex AI Engine --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Agent Type: {args.agent_type.upper()} | Network: {args.network.upper()} | Mode: {args.mode.capitalize()}")
    
    if args.mode == 'tournament':
        from tournament import run_tournament
        run_tournament(args)
    elif args.agent_type == 'a2c':
        from a2c_agent import run_a2c
        run_a2c(args)
    elif args.agent_type == 'mcts':
        from mcts_agent import run_mcts
        run_mcts(args)

    print("----------------------------------------------------")


if __name__ == '__main__':
    main() 