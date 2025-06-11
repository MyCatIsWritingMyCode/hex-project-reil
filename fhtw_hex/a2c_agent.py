import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import argparse
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime

class ActorCritic(nn.Module):
    def __init__(self, board_size, action_space_size):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the flattened size after conv layers
        flattened_size = 32 * board_size * board_size
        
        self.actor_fc = nn.Linear(flattened_size, action_space_size)
        self.critic_fc = nn.Linear(flattened_size, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        
        policy = F.softmax(self.actor_fc(x), dim=-1)
        value = self.critic_fc(x)
        
        return policy, value

def select_action(policy, valid_actions, board_size):
    """
    Selects an action based on the policy and a list of valid actions.
    """
    valid_action_indices = [a[0] * board_size + a[1] for a in valid_actions]
    policy_mask = torch.zeros_like(policy)
    policy_mask[:, valid_action_indices] = 1
    
    masked_policy = policy * policy_mask
    
    # Check if all valid actions have a summed probability of 0.
    # This can happen if the policy network assigns 0 probability to all valid moves.
    if masked_policy.sum().item() < 1e-9:
        # Fallback to a uniform distribution over valid actions to prevent NaNs.
        num_valid_actions = len(valid_action_indices)
        if num_valid_actions == 0:
             # This case should ideally not be reached in a game like Hex before a winner is declared.
             # However, as a safeguard, we can return a random valid action if it ever occurs.
             # But here we'll just create a uniform policy over the mask which is all zeros.
             # A better solution might be to raise an error or handle it gracefully based on game logic.
             # For now, we will create a uniform distribution to avoid crashing.
             masked_policy[:, valid_action_indices] = 1.0
        
        masked_policy[:, valid_action_indices] = 1.0 / num_valid_actions

    # Re-normalize the policy over valid actions.
    # Add a small epsilon for numerical stability before creating the distribution.
    masked_policy /= (masked_policy.sum(dim=-1, keepdim=True) + 1e-9)

    dist = Categorical(probs=masked_policy)
    action_index = dist.sample()
    
    log_prob = dist.log_prob(action_index)
    
    action_coords = (action_index.item() // board_size, action_index.item() % board_size)
    
    return action_coords, log_prob

def get_agent_move(agent, board, player, device, board_size):
    """Function to get the A2C agent's move."""
    board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(device)
    if player == -1:
        board_tensor *= -1  # Agent always sees the board from its own perspective

    # Temporarily create a dummy env to get valid actions
    # This is a bit clunky, but avoids passing the whole env object around
    temp_env = hexPosition(board_size)
    temp_env.board = board
    temp_env.player = player
    valid_actions = temp_env.get_action_space()

    with torch.no_grad():
        policy, _ = agent(board_tensor)
    
    # Use a modified select_action or just take the best action
    # For gameplay, we should be deterministic and pick the best move
    valid_action_indices = [a[0] * board_size + a[1] for a in valid_actions]
    policy_mask = torch.zeros_like(policy.cpu())
    policy_mask[:, valid_action_indices] = 1
    masked_policy = policy.cpu() * policy_mask
    
    # Check if all valid actions have a summed probability of 0.
    if masked_policy.sum().item() < 1e-9:
        # Fallback to random choice among valid actions if policy is all zeros
        from random import choice
        return choice(valid_actions)

    best_action_idx = masked_policy.argmax().item()
    action_coords = (best_action_idx // board_size, best_action_idx % board_size)

    return action_coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C for Hex')
    parser.add_argument('--board-size', type=int, default=7, help='Size of the Hex board')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--n-episodes', type=int, default=50000, help='Number of episodes for training')
    parser.add_argument('--plot-every', type=int, default=100, help='How often to update the plot')
    parser.add_argument('--environment', type=str, default='apple', choices=['windows', 'apple', 'kaggle'], help='The training environment to set the device')
    parser.add_argument('--play', action='store_true', help='Flag to play against the trained model')
    parser.add_argument('--model-path', type=str, default='a2c_hex_agent.pth', help='Path to the saved model weights')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, -1], help='Choose to be player 1 (white) or -1 (black)')
    parser.add_argument('--test-random', action='store_true', help='Test the agent against a random opponent')
    parser.add_argument('--test-episodes', type=int, default=100, help='Number of episodes for testing')

    args = parser.parse_args()

    print(f"--- A2C Hex Agent --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")

    # Set device based on environment
    device = None
    if args.environment == 'apple':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU")
    elif args.environment in ['windows', 'kaggle']:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    
    if device is None:
        device = torch.device("cpu")
        print("Defaulting to CPU")

    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Device: {device}")
    print("---------------------")

    env = hexPosition(args.board_size)
    action_space_size = args.board_size * args.board_size
    agent = ActorCritic(args.board_size, action_space_size).to(device)

    if args.play:
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            agent.eval()
            print(f"Model loaded from {args.model_path}")

            def machine_player(board, action_set):
                # The hex engine passes the board and available moves.
                # We need to know whose turn it is to make the agent see the correct board state.
                # We can infer this from the number of pieces on the board.
                num_white_stones = sum(row.count(1) for row in board)
                num_black_stones = sum(row.count(-1) for row in board)
                player = 1 if num_white_stones == num_black_stones else -1
                return get_agent_move(agent, board, player, device, args.board_size)

            env.human_vs_machine(human_player=args.human_player, machine=machine_player)

        except FileNotFoundError:
            print(f"Error: Model file not found at {args.model_path}. Please train the model first.")
        except Exception as e:
            print(f"An error occurred: {e}")

    elif args.test_random:
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            agent.eval()
            print(f"\nModel loaded from {args.model_path}")
            print(f"--- Running Baseline Test vs. Random Agent ---")

            def agent_player_func(board, action_set):
                num_white_stones = sum(row.count(1) for row in board)
                num_black_stones = sum(row.count(-1) for row in board)
                player = 1 if num_white_stones == num_black_stones else -1
                return get_agent_move(agent, board, player, device, args.board_size)

            # Test as Player 1
            p1_wins = 0
            for _ in trange(args.test_episodes, desc="Agent as P1 (White)"):
                env.reset()
                winner = env.machine_vs_machine_silent(machine1=agent_player_func) # machine2 is random by default
                if winner == 1:
                    p1_wins += 1
            
            # Test as Player 2
            p2_wins = 0
            for _ in trange(args.test_episodes, desc="Agent as P2 (Black)"):
                env.reset()
                winner = env.machine_vs_machine_silent(machine2=agent_player_func) # machine1 is random by default
                if winner == -1:
                    p2_wins += 1

            print("\n--- Test Results ---")
            print(f"Agent as Player 1 (White): {p1_wins}/{args.test_episodes} wins -> {p1_wins/args.test_episodes:.2%}")
            print(f"Agent as Player 2 (Black): {p2_wins}/{args.test_episodes} wins -> {p2_wins/args.test_episodes:.2%}")
            print("--------------------")

        except FileNotFoundError:
            print(f"Error: Model file not found at {args.model_path}. Please train the model first.")
        except Exception as e:
            print(f"An error occurred: {e}")

    else: # Training Mode
        optimizer = optim.Adam(agent.parameters(), lr=args.lr)

        print("\n--- Starting Training ---")
        
        win_rates = []
        p1_wins = 0
        games_played = 0

        for i_episode in trange(args.n_episodes):
            env.reset()
        
            episode_history = []
        
            while env.winner == 0:
                current_player = env.player
                # The board state needs to be on the correct device
                board_tensor = torch.FloatTensor(env.board).unsqueeze(0).unsqueeze(0).to(device)
            
                # Flip board for black player to keep pov consistent for the network
                if current_player == -1:
                    board_tensor *= -1

                policy, value = agent(board_tensor)
            
                valid_actions = env.get_action_space()
                action, log_prob = select_action(policy.cpu(), valid_actions, args.board_size)
            
                env.move(action)
            
                episode_history.append({'log_prob': log_prob, 'value': value, 'player': current_player})

            # Game is over, update stats and calculate returns
            winner = env.winner
            games_played += 1
            if winner == 1:
                p1_wins += 1
        
            if i_episode % args.plot_every == 0 and games_played > 0:
                win_rates.append(p1_wins / games_played)
                p1_wins = 0
                games_played = 0

            returns = []
            T = len(episode_history)
            for t in range(T):
                player_at_t = episode_history[t]['player']
                reward = 1 if player_at_t == winner else -1
                G = (args.gamma ** (T - 1 - t)) * reward
                returns.append(G)
        
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            if T > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            log_probs = torch.cat([h['log_prob'] for h in episode_history]).to(device)
            values = torch.cat([h['value'] for h in episode_history]).squeeze().to(device)
        
            advantages = returns - values
        
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(values, returns)
        
            loss = actor_loss + 0.5 * critic_loss
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("\n--- Training Finished ---")
    
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(win_rates)) * args.plot_every, win_rates)
        plt.title('Player 1 Win Rate Over Training')
        plt.xlabel('Episodes')
        plt.ylabel(f'Win Rate (Avg over {args.plot_every} games)')
        plt.grid(True)
        plt.savefig('a2c_training_progress.png')
        print("Training plot saved to a2c_training_progress.png")

        # Add saving the model
        torch.save(agent.state_dict(), 'a2c_hex_agent.pth')
        print("Model saved to a2c_hex_agent.pth") 