import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Hex AI Engine', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Core arguments
    parser.add_argument('--agent', type=str, default='a2c', choices=['a2c', 'mcts'], help='The type of agent to use.')
    parser.add_argument('--network', type=str, default='resnet', choices=['cnn', 'resnet', 'miniresnet'], help='The network architecture to use.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'test', 'tournament'], help='The mode to run the agent in.')
    parser.add_argument('--board-size', type=int, default=7, help='Size of the Hex board.')
    parser.add_argument('--model-path', type=str, help='Path for the primary model for saving or loading.')
    parser.add_argument('--output-file', type=str, default='a2c_staged_agent.pth', help='Path to save the final trained model.')

    # Training arguments
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument('--n-episodes', type=int, default=5000, help='Total number of episodes for training.')
    train_group.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    train_group.add_argument('--win-rate-threshold', type=float, default=0.85, help='Target win rate to stop a training stage early.')
    train_group.add_argument('--environment', type=str, default='apple', choices=['windows', 'apple', 'kaggle'], help='The training environment to set the device.')

    # A2C specific arguments
    a2c_group = parser.add_argument_group('A2C Arguments')
    a2c_group.add_argument('--a2c-staged-training', action='store_true', help='Enable staged training for the A2C agent against baseline opponents.')
    a2c_group.add_argument('--a2c-gamma', type=float, default=0.99, help='Discount factor for A2C.')
    a2c_group.add_argument('--a2c-update-every-n-episodes', type=int, default=20, help='Number of episodes to collect before a self-play A2C training update.')
    a2c_group.add_argument('--a2c-batch-size', type=int, default=1024, help='Batch size for A2C training updates.')
    a2c_group.add_argument('--p1-win-rate-target', type=float, default=0.55, help='Target win rate for P1 to dynamically adjust P2 oversampling in self-play.')
    a2c_group.add_argument('--dynamic-oversampling-strength', type=float, default=2.0, help='Multiplier for how aggressively to oversample P2 data in self-play.')

    # MCTS specific arguments
    mcts_group = parser.add_argument_group('MCTS Arguments')
    mcts_group.add_argument('--mcts-simulations', type=int, default=50, help='Number of MCTS simulations per move.')
    mcts_group.add_argument('--mcts-epochs', type=int, default=10, help='Number of training epochs per iteration for MCTS.')
    mcts_group.add_argument('--mcts-batch-size', type=int, default=64, help='Batch size for MCTS training.')
    mcts_group.add_argument('--mcts-dynamic-starts', action='store_true', help='Use dynamic starting positions for MCTS training.')
    mcts_group.add_argument('--opponent-type', type=str, default='RandomAgent', choices=['RandomAgent', 'GreedyAgent'], help='Opponent for MCTS training.')
    
    # Play/Test/Tournament arguments
    play_test_group = parser.add_argument_group('Play/Test/Tournament Arguments')
    play_test_group.add_argument('--human-player', type=int, default=1, choices=[1, -1], help='Choose to be player 1 (white) or -1 (black) in play mode.')
    play_test_group.add_argument('--test-episodes', type=int, default=100, help='Number of episodes for testing.')
    play_test_group.add_argument('--p1-agent-type', type=str, default='a2c', choices=['a2c', 'mcts', 'greedy', 'random'], help='Agent type for Player 1 in tournament.')
    play_test_group.add_argument('--p1-network-type', type=str, default='resnet', choices=['cnn', 'resnet', 'miniresnet'], help='Network for Player 1.')
    play_test_group.add_argument('--p1-model-path', type=str, default='a2c_hex_agent.pth', help='Model path for Player 1.')
    play_test_group.add_argument('--p2-agent-type', type=str, default='random', choices=['a2c', 'mcts', 'greedy', 'random'], help='Agent type for Player 2 in tournament.')
    play_test_group.add_argument('--p2-network-type', type=str, default='resnet', choices=['cnn', 'resnet', 'miniresnet'], help='Network for Player 2.')
    play_test_group.add_argument('--p2-model-path', type=str, help='Model path for Player 2.')

    args = parser.parse_args()

    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = f"{args.agent}_hex_agent.pth"

    print(f"--- Hex AI Engine --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    if args.mode == 'tournament':
        print(f"Mode: {args.mode.capitalize()}")
    else:
        print(f"Agent: {args.agent.upper()} | Network: {args.network.upper()} | Mode: {args.mode.capitalize()}")
    
    # Dispatch to the correct handler
    if args.mode == 'tournament':
        print("Dispatching to tournament handler...")
        from tournament import run_tournament
        run_tournament(args)
    elif args.agent == 'a2c':
        print(f"Dispatching to A2C agent handler (mode: {args.mode})...")
        from a2c_agent import run_a2c
        run_a2c(args)
    elif args.agent == 'mcts':
        print(f"Dispatching to MCTS agent handler (mode: {args.mode})...")
        from mcts_agent import run_mcts
        run_mcts(args)

    print("----------------------------------------------------")


if __name__ == '__main__':
    main() 