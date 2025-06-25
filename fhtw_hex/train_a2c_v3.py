import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from networks import ResNet, ActorCritic
import pandas as pd
import os
import argparse
import random
import torch.nn as nn

def generate_plots(loss_hist, p_loss_hist, v_loss_hist, win_rate_hist, filename):
    """Generates and saves training plots."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('A2C Training Progress', fontsize=16)

    # Losses
    axs[0].plot(loss_hist, label='Total Loss', color='r')
    axs[0].plot(p_loss_hist, label='Policy Loss', linestyle='--', alpha=0.7)
    axs[0].plot(v_loss_hist, label='Value Loss', linestyle='--', alpha=0.7)
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Losses')
    axs[0].legend()
    axs[0].grid(True)
    
    # Win Rate
    axs[1].plot(win_rate_hist, label='Win Rate', color='b')
    axs[1].set_xlabel('Log Interval')
    axs[1].set_ylabel('Win Rate')
    axs[1].set_title('Agent Win Rate vs. Self')
    axs[1].axhline(y=0.5, color='gray', linestyle='--')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"Training plot saved to {filename}")

def select_action(policy, valid_actions, board_size):
    """
    Selects an action based on the policy and a list of valid actions.
    """
    valid_action_indices = [a[0] * board_size + a[1] for a in valid_actions]
    policy_mask = torch.zeros_like(policy)
    policy_mask[:, valid_action_indices] = 1
    
    masked_policy = policy * policy_mask
    
    if masked_policy.sum().item() < 1e-9:
        num_valid_actions = len(valid_action_indices)
        if num_valid_actions > 0:
            masked_policy[:, valid_action_indices] = 1.0 / num_valid_actions
        else: # Should not happen in a normal game
            return None, None

    masked_policy /= (masked_policy.sum(dim=-1, keepdim=True) + 1e-9)

    dist = Categorical(probs=masked_policy)
    action_index = dist.sample()
    
    log_prob = dist.log_prob(action_index)
    
    action_coords = (action_index.item() // board_size, action_index.item() % board_size)
    
    return action_coords, log_prob

def get_symmetries(board, policy, board_size):
    """
    Get all 8 symmetries of the board and policy.
    """
    symmetries = []
    # board is (board_size, board_size) numpy array
    # policy is (board_size*board_size,) numpy array
    
    policy_grid = policy.reshape(board_size, board_size)
    
    current_board = board
    current_policy_grid = policy_grid
    
    for _ in range(4):
        # Add original and flipped versions
        symmetries.append((current_board.copy(), current_policy_grid.flatten()))
        symmetries.append((np.fliplr(current_board).copy(), np.fliplr(current_policy_grid).copy().flatten()))
        
        # Rotate 90 degrees
        current_board = np.rot90(current_board)
        current_policy_grid = np.rot90(current_policy_grid)
        
    return symmetries


def run_a2c_augmented_self_play(args):
    """A2C self-play with data augmentation."""
    device = torch.device("mps" if torch.backends.mps.is_available() and args.environment == 'apple' else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Device: {device}")
    print("---------------------")

    env = hexPosition(args.board_size)
    action_space_size = args.board_size * args.board_size
    
    if args.network == 'resnet':
        agent = ResNet(args.board_size, action_space_size).to(device)
        print("Using ResNet architecture.")
    elif args.network == 'cnn':
        agent = ActorCritic(args.board_size, action_space_size).to(device)
        print("Using CNN (ActorCritic) architecture.")
    else:
        raise ValueError(f"Unknown network type: {args.network}")

    if args.model_path and os.path.exists(args.model_path):
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Could not load model from {args.model_path}. Starting from scratch. Error: {e}")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    experience_buffer = []
    running_rewards = []
    total_wins = 0
    total_games = 0
    loss_history = []
    win_rate_history = []
    policy_loss_history = []
    value_loss_history = []

    # Loss function for policy (Kullback-Leibler Divergence)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')

    pbar = trange(args.n_episodes, desc="Self-Play Training")
    for i_episode in pbar:
        env.reset()
        episode_history = []
        
        while env.winner == 0:
            current_player = env.player
            board_state = np.array(env.board)
            
            canonical_board = board_state if current_player == 1 else -board_state
            
            board_tensor = torch.FloatTensor(canonical_board).unsqueeze(0).unsqueeze(0).to(device)
            
            policy_probs, value = agent(board_tensor)
            
            valid_actions = env.get_action_space()
            action, log_prob = select_action(policy_probs.cpu(), valid_actions, args.board_size)

            if action is None: # No valid moves
                break

            episode_history.append({
                'canonical_board': canonical_board,
                'policy': policy_probs.squeeze().cpu().detach().numpy(),
                'log_prob': log_prob, 
                'value': value,
                'player': current_player
            })
            
            env.move(action)

        winner = env.winner
        total_games += 1
        if winner == 1:
            total_wins += 1

        T = len(episode_history)
        for t, step_data in enumerate(episode_history):
            player = step_data['player']
            # Reward is from the perspective of the player at that turn
            reward = winner * player
            
            # Use discounted returns - for simplicity in this AlphaZero-like setup, we use the final game outcome for every state.
            G = reward
            
            # Get symmetries
            board_symmetries = get_symmetries(step_data['canonical_board'], step_data['policy'], args.board_size)
            
            for board_sym, policy_sym in board_symmetries:
                experience_buffer.append({
                    'board': board_sym,
                    'policy': policy_sym,
                    'return': G,
                })

        if len(experience_buffer) >= args.batch_size:
            # Prepare batch for training
            batch = random.sample(experience_buffer, args.batch_size)
            
            boards = torch.FloatTensor(np.array([exp['board'] for exp in batch])).unsqueeze(1).to(device)
            target_policies = torch.FloatTensor(np.array([exp['policy'] for exp in batch])).to(device)
            target_returns = torch.tensor([exp['return'] for exp in batch], dtype=torch.float32).unsqueeze(1).to(device)

            # Forward pass
            pred_policies, pred_values = agent(boards)
            
            # Calculate losses
            # ResNet outputs log_softmax, and target_policies are probabilities. KLDivLoss is appropriate.
            policy_loss = policy_loss_fn(pred_policies, target_policies)
            value_loss = F.mse_loss(pred_values, target_returns)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            policy_loss_history.append(policy_loss.item())
            value_loss_history.append(value_loss.item())
            
            # Clear a portion of the buffer to make space for new experiences
            experience_buffer = experience_buffer[args.batch_size//2:]

        if (i_episode + 1) % args.log_interval == 0:
            win_rate = total_wins / total_games if total_games > 0 else 0
            win_rate_history.append(win_rate)
            avg_loss = np.mean(loss_history[-10:]) if loss_history else 0
            avg_ploss = np.mean(policy_loss_history[-10:]) if policy_loss_history else 0
            avg_vloss = np.mean(value_loss_history[-10:]) if value_loss_history else 0
            pbar.set_postfix({
                "W-Rate": f"{win_rate:.2f}", 
                "Loss": f"{avg_loss:.3f}",
                "P-Loss": f"{avg_ploss:.3f}",
                "V-Loss": f"{avg_vloss:.3f}"
            })
            total_wins = 0
            total_games = 0

        if (i_episode + 1) % args.save_interval == 0:
            if args.output_file:
                save_path = f"{os.path.splitext(args.output_file)[0]}_e{i_episode+1}.pth"
                torch.save(agent.state_dict(), save_path)
                print(f"\nModel saved to {save_path}")

    if args.output_file:
        torch.save(agent.state_dict(), args.output_file)
        print(f"Final model saved to {args.output_file}")
        
    if args.plot_file:
        generate_plots(loss_history, policy_loss_history, value_loss_history, win_rate_history, args.plot_file)
        print(f"Training plot saved to {args.plot_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A2C Augmented Self-Play Trainer")
    parser.add_argument("--board_size", type=int, default=5, help="Size of the hex board")
    parser.add_argument("--network", type=str, default="resnet", choices=["resnet", "cnn"], help="Network architecture to use")
    parser.add_argument("--n_episodes", type=int, default=50000, help="Number of episodes for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging training progress")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval for saving the model")
    parser.add_argument("--output_file", type=str, default="a2c_v3_agent.pth", help="Path to save the trained model")
    parser.add_argument("--model_path", type=str, help="Path to a pre-trained model to continue training")
    parser.add_argument("--plot_file", type=str, default="a2c_v3_training_progress.png", help="Path to save the training plot")
    parser.add_argument("--environment", type=str, default="default", help="Environment for device selection ('apple' for mps)")
    
    args = parser.parse_args()
    run_a2c_augmented_self_play(args) 