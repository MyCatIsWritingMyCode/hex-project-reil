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
    masked_policy /= masked_policy.sum(dim=-1, keepdim=True) # Re-normalize

    dist = Categorical(masked_policy)
    action_index = dist.sample()
    
    log_prob = dist.log_prob(action_index)
    
    action_coords = (action_index.item() // board_size, action_index.item() % board_size)
    
    return action_coords, log_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C for Hex')
    parser.add_argument('--board-size', type=int, default=3, help='Size of the Hex board')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--n-episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--plot-every', type=int, default=100, help='How often to update the plot')
    parser.add_argument('--environment', type=str, default='apple', choices=['windows', 'apple', 'kaggle'], help='The training environment to set the device')
    
    args = parser.parse_args()

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

    env = hexPosition(args.board_size)
    action_space_size = args.board_size * args.board_size
    agent = ActorCritic(args.board_size, action_space_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    print("Starting Training...")
    
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

    print("Training Finished.")
    
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