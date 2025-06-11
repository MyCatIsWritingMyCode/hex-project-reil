import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from networks import ActorCritic, ResNet

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

def run_a2c(args):
    """The main entry point for running the A2C agent."""
    
    # Set device based on environment
    device = None
    if args.environment == 'apple':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif args.environment in ['windows', 'kaggle']:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    if device is None:
        device = torch.device("cpu")
        
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Device: {device}")
    print("---------------------")

    env = hexPosition(args.board_size)
    action_space_size = args.board_size * args.board_size
    
    if args.network == 'cnn':
        agent = ActorCritic(args.board_size, action_space_size).to(device)
    elif args.network == 'resnet':
        # A2C is not optimized for ResNet, but we allow it for consistency.
        # The user should be aware that this combination might not be ideal.
        print("Warning: Using ResNet with the A2C agent may not be optimal.")
        agent = ResNet(args.board_size, action_space_size).to(device)
    else:
        raise ValueError(f"Unknown network type: {args.network}")

    if args.mode == 'play' or args.mode == 'test':
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            agent.eval()
            print(f"\nModel loaded from {args.model_path}")

            def agent_player_func(board, action_set):
                num_white_stones = sum(row.count(1) for row in board)
                num_black_stones = sum(row.count(-1) for row in board)
                player = 1 if num_white_stones == num_black_stones else -1
                return get_agent_move(agent, board, player, device, args.board_size)

            if args.mode == 'play':
                 env.human_vs_machine(human_player=args.human_player, machine=agent_player_func)
            
            if args.mode == 'test':
                print(f"--- Running Baseline Test vs. Random Agent ---")
                # Test as Player 1
                p1_wins = 0
                for _ in trange(args.test_episodes, desc="Agent as P1 (White)"):
                    env.reset()
                    winner = env.machine_vs_machine_silent(machine1=agent_player_func)
                    if winner == 1:
                        p1_wins += 1
                
                # Test as Player 2
                p2_wins = 0
                for _ in trange(args.test_episodes, desc="Agent as P2 (Black)"):
                    env.reset()
                    winner = env.machine_vs_machine_silent(machine2=agent_player_func)
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

    elif args.mode == 'train':
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

        print("\n--- Starting Training ---")
        
        win_rates = []
        p1_wins = 0
        games_played = 0
        plot_every = 100 # Could be an arg

        for i_episode in trange(args.n_episodes, desc="Training A2C"):
            env.reset()
            episode_history = []
            
            while env.winner == 0:
                current_player = env.player
                board_tensor = torch.FloatTensor(env.board).unsqueeze(0).unsqueeze(0).to(device)
                
                if current_player == -1:
                    board_tensor *= -1

                policy, value = agent(board_tensor)
                
                valid_actions = env.get_action_space()
                action, log_prob = select_action(policy.cpu(), valid_actions, args.board_size)
                
                env.move(action)
                
                episode_history.append({'log_prob': log_prob, 'value': value, 'player': current_player})

            winner = env.winner
            games_played += 1
            if winner == 1:
                p1_wins += 1
        
            if i_episode % plot_every == 0 and games_played > 0:
                win_rates.append(p1_wins / games_played)
                p1_wins = 0
                games_played = 0

            returns = []
            T = len(episode_history)
            for t in range(T):
                player_at_t = episode_history[t]['player']
                reward = 1 if player_at_t == winner else -1
                G = (args.a2c_gamma ** (T - 1 - t)) * reward
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
    
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(win_rates)) * plot_every, win_rates)
        plt.title('A2C Player 1 Win Rate Over Training')
        plt.xlabel('Episodes')
        plt.ylabel(f'Win Rate (Avg over {plot_every} games)')
        plt.grid(True)
        plt.savefig('a2c_training_progress.png')
        print("Training plot saved to a2c_training_progress.png")

        torch.save(agent.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}") 