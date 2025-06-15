import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from networks import ActorCritic, ResNet
import pandas as pd
from plotting import generate_training_plots, generate_staged_training_plots
from baseline_agents import RandomAgent, GreedyAgent, DefensiveAgent, AggressiveAgent
import random
import os

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
    if args.a2c_staged_training:
        run_a2c_staged_training(args)
    else:
        run_a2c_self_play(args)

def run_a2c_staged_training(args):
    """Run A2C training in stages against a pool of baseline opponents."""
    
    device = torch.device("mps" if torch.backends.mps.is_available() and args.environment == 'apple' else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    action_space_size = args.board_size * args.board_size
    if args.network == 'cnn':
        agent = ActorCritic(args.board_size, action_space_size).to(device)
    elif args.network == 'resnet':
        agent = ResNet(args.board_size, action_space_size).to(device)
    else:
        raise ValueError(f"Unknown network type: {args.network}")

    if args.model_path and os.path.exists(args.model_path):
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Could not load model from {args.model_path}. Starting from scratch. Error: {e}")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    opponent_pool = ['RandomAgent', 'GreedyAgent', 'AggressiveAgent', 'DefensiveAgent', 'Mixed']
    all_wins = []
    all_losses = []

    for stage, opponent_name in enumerate(opponent_pool):
        print(f"\n--- Stage {stage + 1}: Training against {opponent_name} ---")
        
        experience_buffer = []
        win_history = []
        consecutive_high_win_rate = 0

        opponent_list = ['RandomAgent', 'GreedyAgent', 'AggressiveAgent', 'DefensiveAgent']
        
        stage_episodes = args.n_episodes // len(opponent_pool)

        for i_episode in trange(stage_episodes, desc=f"Stage {stage+1}/{len(opponent_pool)} vs {opponent_name}"):
            env = hexPosition(args.board_size)
            episode_history = []

            if opponent_name == 'Mixed':
                current_opponent_name = random.choice(opponent_list)
                opponent = get_opponent(current_opponent_name)
            else:
                opponent = get_opponent(opponent_name)
            
            a2c_player = 1 if i_episode % 2 == 0 else -1

            while env.winner == 0:
                current_player = env.player
                
                if current_player == a2c_player:
                    board_tensor = torch.FloatTensor(env.board).unsqueeze(0).unsqueeze(0).to(device)
                    if current_player == -1:
                        board_tensor *= -1

                    policy, value = agent(board_tensor)
                    valid_actions = env.get_action_space()
                    action, log_prob = select_action(policy.cpu(), valid_actions, args.board_size)
                    
                    episode_history.append({'log_prob': log_prob, 'value': value})
                    env.move(action)
                else:
                    action = opponent.select_move(env.board, env.get_action_space(), current_player)
                    env.move(action)

            winner = env.winner
            win_history.append(1 if winner == a2c_player else 0)
            all_wins.append(1 if winner == a2c_player else 0)

            returns = []
            T = len(episode_history)
            for t in range(T):
                reward = 1 if a2c_player == winner else -1
                G = (args.a2c_gamma ** (T - 1 - t)) * reward
                returns.append(G)

            if T > 1:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
            
            for t in range(T):
                experience_buffer.append({
                    'log_prob': episode_history[t]['log_prob'],
                    'value': episode_history[t]['value'],
                    'return': returns_tensor[t]
                })

            if len(experience_buffer) >= args.a2c_batch_size:
                log_probs = torch.cat([exp['log_prob'] for exp in experience_buffer]).to(device)
                values = torch.cat([exp['value'] for exp in experience_buffer]).squeeze()
                returns = torch.tensor([exp['return'] for exp in experience_buffer]).to(device)

                advantages = returns - values.detach()
                actor_loss = -(log_probs * advantages).mean()
                critic_loss = F.mse_loss(values, returns)

                loss = actor_loss + critic_loss
                all_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                experience_buffer = []

            if len(win_history) > 100:
                recent_win_rate = sum(win_history[-100:]) / 100
                if recent_win_rate >= args.win_rate_threshold:
                    consecutive_high_win_rate += 1
                else:
                    consecutive_high_win_rate = 0
                
                if consecutive_high_win_rate >= 10:
                    print(f"Win rate threshold reached for 10 consecutive checks. Moving to next stage.")
                    break
        
        # --- End of Stage: Perform a final update on any remaining experience ---
        if len(experience_buffer) > 0:
            log_probs = torch.cat([exp['log_prob'] for exp in experience_buffer]).to(device)
            values = torch.cat([exp['value'] for exp in experience_buffer]).squeeze()
            returns = torch.tensor([exp['return'] for exp in experience_buffer]).to(device)

            advantages = returns - values.detach()
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + critic_loss
            all_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            experience_buffer = [] # Clear for next stage
    
    print("\n--- Staged Training Complete ---")
    if args.output_file:
        torch.save(agent.state_dict(), args.output_file)
        print(f"Model saved to {args.output_file}")

    generate_staged_training_plots(all_losses, all_wins, "a2c_staged_training_progress.png")
    
def run_a2c_self_play(args):
    """Original self-play training logic for A2C."""
    device = torch.device("mps" if torch.backends.mps.is_available() and args.environment == 'apple' else "cuda" if torch.cuda.is_available() else "cpu")
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

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    print("\n--- Starting Training (A2C v2 Self-Play) ---")
    
    training_log = []
    experience_buffer = []
    p1_wins_in_batch = 0
    games_in_batch = 0

    for i_episode in trange(args.n_episodes, desc="Training A2C Self-Play"):
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
        
        returns = torch.tensor(returns, dtype=torch.float32)
        if T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for t in range(T):
            board_state = np.copy(episode_history[t]['board_state_before_move'])
            if episode_history[t]['player'] == -1:
                board_state *= -1 
                
            experience_buffer.append({
                'log_prob': episode_history[t]['log_prob'],
                'value': episode_history[t]['value'],
                'return': returns[t],
                'player': episode_history[t]['player'],
                'board_state': board_state
            })
        
        if games_in_batch >= args.a2c_update_every_n_episodes:
            p1_win_rate = p1_wins_in_batch / games_in_batch if games_in_batch > 0 else 0
            p2_oversampling_ratio = 1.0 + args.dynamic_oversampling_strength * (p1_win_rate - args.p1_win_rate_target)
            p2_oversampling_ratio = max(0.1, p2_oversampling_ratio)

            p1_exp = [exp for exp in experience_buffer if exp['player'] == 1]
            p2_exp = [exp for exp in experience_buffer if exp['player'] == -1]
            
            if not p1_exp or not p2_exp:
                experience_buffer.clear()
                p1_wins_in_batch = 0
                games_in_batch = 0
                continue

            batch_size = len(experience_buffer)
            p2_batch_size = int(batch_size * (p2_oversampling_ratio / (1 + p2_oversampling_ratio)))
            p1_batch_size = batch_size - p2_batch_size

            p1_sample_idx = np.random.randint(len(p1_exp), size=p1_batch_size)
            p2_sample_idx = np.random.randint(len(p2_exp), size=p2_batch_size)

            p1_sample = [p1_exp[i] for i in p1_sample_idx]
            p2_sample = [p2_exp[i] for i in p2_sample_idx]
            
            batch = p1_sample + p2_sample
            
            log_probs = torch.cat([h['log_prob'] for h in batch]).to(device)
            values = torch.cat([h['value'] for h in batch]).squeeze().to(device)
            returns_batch = torch.tensor([h['return'] for h in batch], dtype=torch.float32).to(device)
            
            boards_for_entropy = [exp['board_state'] for exp in batch]
            boards_tensor = torch.FloatTensor(np.array(boards_for_entropy)).unsqueeze(1).to(device)
            with torch.no_grad():
                policy_batch, _ = agent(boards_tensor)
            
            entropy_dist = Categorical(probs=policy_batch)
            policy_entropy = entropy_dist.entropy().mean().item()

            advantages = returns_batch - values
            
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(values, returns_batch)
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_log.append({
                'episode': i_episode,
                'p1_win_rate_batch': p1_win_rate,
                'p2_oversampling_ratio': p2_oversampling_ratio,
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'total_loss': loss.item(),
                'policy_entropy': policy_entropy
            })

            experience_buffer.clear()
            p1_wins_in_batch = 0
            games_in_batch = 0

    print("\n--- Training Finished ---")
    
    if training_log:
        log_df = pd.DataFrame(training_log)
        log_df.to_csv('a2c_training_log.csv', index=False)
        print("Detailed training log saved to a2c_training_log.csv")
        # Assuming generate_training_plots is adapted for DataFrame input
        # generate_training_plots(log_df, 'A2C Self-Play', args.p1_win_rate_target)

    if args.output_file:
        torch.save(agent.state_dict(), args.output_file)
        print(f"Model saved to {args.output_file}")

def get_opponent(opponent_type):
    """Factory function to get an opponent agent."""
    if opponent_type == 'GreedyAgent':
        return GreedyAgent()
    elif opponent_type == 'RandomAgent':
        return RandomAgent()
    elif opponent_type == 'DefensiveAgent':
        return DefensiveAgent()
    elif opponent_type == 'AggressiveAgent':
        return AggressiveAgent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}") 