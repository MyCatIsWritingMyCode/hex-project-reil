import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from networks import ActorCritic, ResNet
import pandas as pd
from plotting import generate_training_plots

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

        print("\n--- Starting Training (A2C v2) ---")
        
        # --- Set up logging ---
        training_log = []
        experience_buffer = []
        p1_wins_in_batch = 0
        games_in_batch = 0

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
                board_state_before_move = np.copy(env.board)
                env.move(action)
                episode_history.append({
                    'log_prob': log_prob, 
                    'value': value, 
                    'player': current_player, 
                    'board_state_before_move': board_state_before_move
                })

            # --- Store episode results ---
            winner = env.winner
            games_in_batch += 1
            if winner == 1:
                p1_wins_in_batch += 1
            
            returns = []
            T = len(episode_history)
            for t in range(T):
                player_at_t = episode_history[t]['player']
                reward = 1 if player_at_t == winner else -1
                G = (args.a2c_gamma ** (T - 1 - t)) * reward
                returns.append(G)
            
            # Normalize returns for this episode
            returns = torch.tensor(returns, dtype=torch.float32)
            if T > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            for t in range(T):
                board_state = np.copy(episode_history[t]['board_state_before_move'])
                if episode_history[t]['player'] == -1:
                    board_state *= -1 # Canonical form
                    
                experience_buffer.append({
                    'log_prob': episode_history[t]['log_prob'],
                    'value': episode_history[t]['value'],
                    'return': returns[t],
                    'player': episode_history[t]['player'],
                    'board_state': board_state
                })
            
            # --- Perform a training update if the batch is full ---
            if games_in_batch >= args.a2c_update_every_n_episodes:
                
                # --- Dynamic Oversampling Calculation ---
                p1_win_rate = p1_wins_in_batch / games_in_batch if games_in_batch > 0 else 0
                p2_oversampling_ratio = 1.0 + args.dynamic_oversampling_strength * (p1_win_rate - args.p1_win_rate_target)
                p2_oversampling_ratio = max(0.1, p2_oversampling_ratio)

                # --- Separate data for sampling ---
                p1_exp = [exp for exp in experience_buffer if exp['player'] == 1]
                p2_exp = [exp for exp in experience_buffer if exp['player'] == -1]
                
                if not p1_exp or not p2_exp: # Skip if one player had no moves
                    experience_buffer.clear()
                    p1_wins_in_batch = 0
                    games_in_batch = 0
                    continue

                # --- Construct Training Batch ---
                batch_size = len(experience_buffer)
                p2_batch_size = int(batch_size * (p2_oversampling_ratio / (1 + p2_oversampling_ratio)))
                p1_batch_size = batch_size - p2_batch_size

                p1_sample_idx = np.random.randint(len(p1_exp), size=p1_batch_size)
                p2_sample_idx = np.random.randint(len(p2_exp), size=p2_batch_size)

                p1_sample = [p1_exp[i] for i in p1_sample_idx]
                p2_sample = [p2_exp[i] for i in p2_sample_idx]
                
                batch = p1_sample + p2_sample
                
                # --- Perform Update ---
                log_probs = torch.cat([h['log_prob'] for h in batch]).to(device)
                values = torch.cat([h['value'] for h in batch]).squeeze().to(device)
                returns_batch = torch.tensor([h['return'] for h in batch], dtype=torch.float32).to(device)
                
                # --- Calculate Entropy ---
                # Re-evaluate policy for the batch to calculate current entropy
                boards_for_entropy = [exp['board_state'] for exp in batch]
                boards_tensor = torch.FloatTensor(np.array(boards_for_entropy)).unsqueeze(1).to(device)
                with torch.no_grad():
                    policy_batch, _ = agent(boards_tensor)
                
                # Use the distribution from select_action to get entropy
                entropy_dist = Categorical(probs=policy_batch)
                policy_entropy = entropy_dist.entropy().mean().item()

                advantages = returns_batch - values
                
                actor_loss = -(log_probs * advantages.detach()).mean()
                critic_loss = F.mse_loss(values, returns_batch)
                loss = actor_loss + 0.5 * critic_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # --- Log metrics for this batch ---
                training_log.append({
                    'episode': i_episode,
                    'p1_win_rate_batch': p1_win_rate,
                    'p2_oversampling_ratio': p2_oversampling_ratio,
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'total_loss': loss.item(),
                    'policy_entropy': policy_entropy
                })

                # --- Clear buffers for next batch ---
                experience_buffer.clear()
                p1_wins_in_batch = 0
                games_in_batch = 0

        print("\n--- Training Finished ---")
        
        # --- Save detailed log and generate plots ---
        if training_log:
            log_df = pd.DataFrame(training_log)
            log_df.to_csv('a2c_training_log.csv', index=False)
            print("Detailed training log saved to a2c_training_log.csv")
            generate_training_plots(log_df, 'A2C', args.p1_win_rate_target)

        torch.save(agent.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}") 