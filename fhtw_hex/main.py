import argparse
from datetime import datetime
from plotting import generate_staged_training_plots
from a2c_agent import run_a2c_self_play, run_a2c_staged_training
from tournament import run_tournament
from pretrain_mcts import run_mcts_pretraining

def main(args):
    """The main entry point of the application."""
    if args.mode == 'a2c-self-play':
        run_a2c_self_play(args)
    elif args.mode == 'a2c-staged-training':
        run_a2c_staged_training(args)
    elif args.mode == 'tournament':
        run_tournament(args)
    elif args.mode == 'mcts-pretrain':
        run_mcts_pretraining(args)
    else:
        print(f"FATAL: Unknown mode '{args.mode}'.")
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A2C and MCTS agents for Hex.")
    
    # Core settings
    parser.add_argument("--mode", type=str, default="a2c-staged-training", 
                        choices=["a2c-self-play", "a2c-staged-training", "tournament", "mcts-pretrain"], 
                        help="The mode to run the script in.")
    parser.add_argument("--board-size", type=int, default=7, help="Size of the Hex board.")
    parser.add_argument("--environment", type=str, default="apple", 
                        choices=["apple", "windows", "linux"], 
                        help="Specify the environment for device allocation.")

    # A2C settings
    parser.add_argument("--a2c-episodes", type=int, default=25000, help="Number of episodes for A2C training.")
    parser.add_argument("--a2c-batch-size", type=int, default=128, help="Batch size for A2C updates.")
    parser.add_argument("--a2c-learning-rate", type=float, default=1e-4, help="Learning rate for the A2C agent.")
    parser.add_argument("--a2c-model-path", type=str, default="a2c_cnn_teacher.pth", help="Path to save/load the A2C model.")
    parser.add_argument("--a2c-load-model", action="store_true", help="Flag to load a pre-trained A2C model.")
    parser.add_argument("--a2c-staged-training-model-path", type=str, default="a2c_cnn_teacher.pth", help="Path to save the final staged training model.")

    # MCTS settings
    parser.add_argument("--mcts-episodes", type=int, default=1000, help="Number of episodes for MCTS training.")
    parser.add_argument("--mcts-simulations", type=int, default=100, help="Number of MCTS simulations per move.")
    parser.add_argument("--mcts-learning-rate", type=float, default=0.001, help="Learning rate for the MCTS network.")
    parser.add_argument("--mcts-model-path", type=str, default="mcts_agent.pth", help="Path to save/load the MCTS model.")
    parser.add_argument("--mcts-load-model", action="store_true", help="Flag to load a pre-trained MCTS model.")
    
    # MCTS Pre-training settings
    pretrain_group = parser.add_argument_group('MCTS Pre-training')
    pretrain_group.add_argument("--teacher-model-path", type=str, default="a2c_cnn_teacher.pth", help="Path to the teacher A2C model for pre-training.")
    pretrain_group.add_argument("--output-model-path", type=str, default="mcts_pretrained.pth", help="Path to save the pre-trained MCTS model.")
    pretrain_group.add_argument("--num-games", type=int, default=500, help="Number of games to generate for the expert dataset.")
    pretrain_group.add_argument("--epochs", type=int, default=20, help="Number of epochs for pre-training.")
    pretrain_group.add_argument("--batch-size", type=int, default=64, help="Batch size for pre-training.")
    pretrain_group.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for pre-training.")

    # Tournament settings
    parser.add_argument("--test-episodes", type=int, default=100, help="Number of games to play against each opponent in tournament mode.")
    parser.add_argument("--p1-agent-type", type=str, default="a2c", choices=["a2c", "mcts"], help="Type of agent for Player 1 in tournament.")
    parser.add_argument("--p1-network-type", type=str, default="cnn", choices=["cnn", "resnet"], help="Network architecture for Player 1.")
    parser.add_argument("--p1-model-path", type=str, default="a2c_cnn_teacher.pth", help="Path to the model file for Player 1.")
    parser.add_argument("--p2-agent-type", type=str, default=None, choices=["a2c", "mcts", "random", "greedy", "aggressive", "defensive"], help="Type of agent for Player 2 in tournament.")
    parser.add_argument("--p2-network-type", type=str, default=None, choices=["cnn", "resnet"], help="Network architecture for Player 2, if applicable.")
    parser.add_argument("--p2-model-path", type=str, default=None, help="Path to the model file for Player 2, if applicable.")

    args = parser.parse_args()
    #Correct board_size name
    if hasattr(args, 'board_size'):
        pass
    elif hasattr(args, 'board-size'):
        args.board_size = getattr(args, 'board-size')

    main(args)