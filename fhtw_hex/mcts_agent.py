import torch
import torch.nn.functional as F
import numpy as np
import argparse
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime
import math
import os
from copy import deepcopy
from networks import ActorCritic, ResNet, MiniResNet
import pandas as pd
from plotting import generate_mcts_plots
from torch.distributions import Categorical
from baseline_agents import GreedyAgent, AggressiveAgent, DefensiveAgent, RandomAgent

class Node:
    """A node in the Monte Carlo Tree Search tree."""
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # A map from action to Node
        self.N = 0  # Visit count
        self.Q = 0  # Total action value
        self.P = prior_p # Prior probability

class MCTS:
    """The main MCTS class that orchestrates the search."""
    def __init__(self, agent, c_puct=1.0, device='cpu'):
        self.agent = agent
        self.c_puct = c_puct # Exploration-exploitation constant
        self.device = device

    def search(self, env, num_simulations):
        """
        Run the MCTS search for a specified number of simulations.
        """
        root = Node(prior_p=1.0)

        for _ in range(num_simulations):
            node = root
            # Use the new, fast clone method
            sim_env = env.clone()

            # 1. Selection
            while node.children:
                action, node = self._select_child(node)
                sim_env.move(action)

            # 2. Expansion & Evaluation
            # Use the network to get policy and value for the leaf node
            board_tensor = torch.FloatTensor(sim_env.board).unsqueeze(0).unsqueeze(0).to(self.device)
            if sim_env.player == -1:
                board_tensor *= -1

            with torch.no_grad():
                policy, value = self.agent(board_tensor)
            
            # FIXED: Convert log probabilities to probabilities
            policy = torch.exp(policy)  # Convert log_softmax to softmax
            
            value = value.item()

            # The game might be over at this leaf node
            if sim_env.winner != 0:
                # The value should be from the perspective of the current player at the leaf
                if sim_env.player == sim_env.winner:
                    value = 1.0
                elif sim_env.player == -sim_env.winner:
                    value = -1.0
                else: # Draw, not possible in Hex
                    value = 0.0
            else:
                # Expand the node
                valid_actions = sim_env.get_action_space()
                action_probs = policy.squeeze().cpu().numpy()
                board_size = sim_env.size
                
                for action in valid_actions:
                    action_scalar = action[0] * board_size + action[1]
                    node.children[action] = Node(parent=node, prior_p=action_probs[action_scalar])

            # 3. Backpropagation
            current_value = value
            while node is not None:
                node.N += 1
                # FIXED: Q should accumulate values, then we average in UCT
                node.Q += current_value
                # Flip value for parent (opponent's perspective)
                current_value = -current_value
                node = node.parent
        
        return root

    def _select_child(self, node):
        """Select the child with the highest UCT value."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = self._uct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _uct_score(self, parent, child):
        """Upper Confidence Bound for Trees formula."""
        q_value = child.Q / max(child.N, 1)  # Average Q-value, avoid division by zero
        
        exploration_term = self.c_puct * child.P * math.sqrt(parent.N) / (1 + child.N)
        return q_value + exploration_term

    def get_action_probs(self, env, num_simulations, temp=1.0):
        """
        Run MCTS search and return the improved policy (action probabilities).
        """
        root = self.search(env, num_simulations)
        
        counts = {action: node.N for action, node in root.children.items()}
        
        if not counts:
            return {}

        if temp == 0:
            # Deterministic: choose the best action
            best_action = max(counts, key=counts.get)
            probs = {action: 1.0 if action == best_action else 0.0 for action in counts}
        else:
            # Sample from the distribution defined by visit counts
            count_values = np.power(list(counts.values()), 1.0/temp)
            total = sum(count_values)
            if total == 0: # handle case where all counts are zero
                num_valid = len(counts)
                return {action: 1.0/num_valid for action in counts}
            probs = {action: count** (1.0/temp) / total for action, count in counts.items()}
            
        return probs

def execute_episode(agent, mcts, args):
    """Execute a single episode of self-play."""
    env = hexPosition(args.board_size)
    train_examples = []
    
    while env.winner == 0:
        # Get the improved policy from MCTS
        action_probs = mcts.get_action_probs(env, args.mcts_simulations)
        
        # Create a canonical board representation
        canonical_board = np.array(env.board)
        if env.player == -1:
            canonical_board *= -1

        # Store the training example
        # The policy target is a vector of probabilities for all possible actions
        pi = np.zeros(args.board_size**2)
        for action, prob in action_probs.items():
            pi[action[0] * args.board_size + action[1]] = prob
        
        train_examples.append([canonical_board, env.player, pi])

        # Sample an action from the MCTS policy
        action = list(action_probs.keys())[np.random.choice(len(action_probs), p=list(action_probs.values()))]
        env.move(action)

    # Return training examples and the winner of the game
    winner = env.winner
    # FIXED: Value assignment must match canonical board representation
    # Since we always flip the board to player 1's perspective, values should be from player 1's perspective
    examples = []
    for board, player, policy in train_examples:
        # The canonical board is always from player 1's perspective
        # So the value should be +1 if player 1 wins, -1 if player -1 wins
        if winner == 1:
            value = 1.0  # Player 1 won
        elif winner == -1:
            value = -1.0  # Player -1 won
        else:  # Draw (shouldn't happen in Hex)
            value = 0.0
        examples.append((board, player, policy, value))
    return examples, winner

def execute_episode_vs_opponent(agent, mcts, opponent, mcts_player, args):
    """
    Execute a single episode of play against a fixed opponent, with the MCTS agent
    playing as the specified player.
    """
    env = hexPosition(args.board_size)
    train_examples = []
    
    # --- Dynamic Starting Position Logic ---
    if args.mcts_dynamic_starts and np.random.rand() < 0.5: # 50% chance for a dynamic start
        num_random_moves = np.random.randint(1, 4) # 1 to 3 random moves
        for _ in range(num_random_moves):
            if env.winner == 0:
                valid_actions = env.get_action_space()
                action = valid_actions[np.random.randint(len(valid_actions))]
                env.move(action)
    # ------------------------------------

    while env.winner == 0:
        current_player = env.player
        if current_player == mcts_player:
            # MCTS agent's turn
            action_probs = mcts.get_action_probs(env, args.mcts_simulations)
            
            # Create a canonical board representation for the network
            canonical_board = np.array(env.board)
            if current_player == -1:
                canonical_board *= -1

            # Store the training example
            pi = np.zeros(args.board_size**2)
            for action, prob in action_probs.items():
                pi[action[0] * args.board_size + action[1]] = prob
            
            train_examples.append([canonical_board, current_player, pi])

            # Sample an action from the MCTS policy
            action = list(action_probs.keys())[np.random.choice(len(action_probs), p=list(action_probs.values()))]
            env.move(action)
        else:
            # Opponent's turn
            action = opponent.select_move(env.board, env.get_action_space(), current_player)
            env.move(action)

    winner = env.winner
    # FIXED: Value assignment must match the canonical board representation
    # Since we always flip the board to player 1's perspective, values should be from player 1's perspective
    examples = []
    for board, player, policy in train_examples:
        # The canonical board is always from player 1's perspective
        # So the value should be +1 if player 1 wins, -1 if player -1 wins
        if winner == 1:
            value = 1.0  # Player 1 won
        elif winner == -1:
            value = -1.0  # Player -1 won
        else:  # Draw (shouldn't happen in Hex)
            value = 0.0
        examples.append((board, player, policy, value))
    return examples, winner

def get_opponent(opponent_type):
    """Factory function to get an opponent agent."""
    if opponent_type == 'GreedyAgent':
        return GreedyAgent()
    elif opponent_type == 'RandomAgent':
        return RandomAgent()
    elif opponent_type == 'AggressiveAgent':
        return AggressiveAgent()
    elif opponent_type == 'DefensiveAgent':
        return DefensiveAgent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

def run_mcts(args):
    """The main entry point for running the MCTS agent."""
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() and args.environment == 'apple' else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup network
    action_space_size = args.board_size**2
    if args.network == 'cnn':
        agent = ActorCritic(args.board_size, action_space_size).to(device)
    elif args.network == 'resnet':
        agent = ResNet(args.board_size, action_space_size).to(device)
    elif args.network == 'miniresnet':
        agent = MiniResNet(args.board_size, action_space_size).to(device)
    else:
        raise ValueError(f"Unknown network type: {args.network}")
    
    # Load an existing model if provided
    if args.model_path and os.path.exists(args.model_path):
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Could not load model from {args.model_path}. Starting from scratch. Error: {e}")

    # Setup MCTS
    mcts = MCTS(agent, device=device)

    # --- Training Mode ---
    if args.mode == 'train':
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
        
        overall_win_history = []
        overall_loss_history = []

        if args.staged_training:
            print("\n--- Starting Staged MCTS Training ---")
            stages = [
                {'name': 'Greedy', 'episodes': args.stage1_episodes, 'opponent': GreedyAgent()},
                {'name': 'Aggressive', 'episodes': args.stage2_episodes, 'opponent': AggressiveAgent()},
                {'name': 'Defensive', 'episodes': args.stage3_episodes, 'opponent': DefensiveAgent()},
            ]
            baseline_pool = [GreedyAgent(), AggressiveAgent(), DefensiveAgent(), RandomAgent()]

            for stage in stages:
                print(f"\n--- Training Stage: {stage['name']} ({stage['episodes']} episodes) ---")
                stage_examples = []
                stage_wins = []
                for i in trange(1, stage['episodes'] + 1, desc=f"Stage: {stage['name']}"):
                    mcts_player = 1 if i % 2 == 0 else -1
                    new_examples, winner = execute_episode_vs_opponent(agent, mcts, stage['opponent'], mcts_player, args)
                    stage_examples.extend(new_examples)
                    
                    if winner == mcts_player: stage_wins.append(1)
                    else: stage_wins.append(0)
                    
                    if i % args.mcts_epochs == 0 and stage_examples:
                        # --- Training Update ---
                        agent.train()
                        dataset = torch.utils.data.TensorDataset(
                            torch.FloatTensor(np.array([e[0] for e in stage_examples])),
                            torch.FloatTensor(np.array([e[2] for e in stage_examples])),
                            torch.FloatTensor(np.array([e[3] for e in stage_examples])).unsqueeze(1)
                        )
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.mcts_batch_size, shuffle=True)
                        for boards, pis, vs in dataloader:
                            boards, pis, vs = boards.to(device), pis.to(device), vs.to(device)
                            boards = boards.unsqueeze(1); log_probs, value = agent(boards)
                            loss = -torch.sum(pis * log_probs, dim=1).mean() + F.mse_loss(value, vs)
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                        stage_examples = [] # Clear memory

                overall_win_history.extend(stage_wins)
                win_rate = sum(stage_wins) / len(stage_wins) if stage_wins else 0
                print(f"Stage '{stage['name']}' complete. Win Rate: {win_rate:.2f}")

            # Final stage: Mixed Pool
            print(f"\n--- Training Stage: Mixed Pool ({args.mixed_pool_episodes} episodes) ---")
            stage_examples = []
            stage_wins = []
            for i in trange(1, args.mixed_pool_episodes + 1, desc="Stage: Mixed Pool"):
                opponent = baseline_pool[i % len(baseline_pool)]
                mcts_player = 1 if i % 2 == 0 else -1
                new_examples, winner = execute_episode_vs_opponent(agent, mcts, opponent, mcts_player, args)
                stage_examples.extend(new_examples)
                
                if winner == mcts_player: stage_wins.append(1)
                else: stage_wins.append(0)

                if i % args.mcts_epochs == 0 and stage_examples:
                    # --- Training Update ---
                    agent.train()
                    dataset = torch.utils.data.TensorDataset(
                        torch.FloatTensor(np.array([e[0] for e in stage_examples])),
                        torch.FloatTensor(np.array([e[2] for e in stage_examples])),
                        torch.FloatTensor(np.array([e[3] for e in stage_examples])).unsqueeze(1)
                    )
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.mcts_batch_size, shuffle=True)
                    for boards, pis, vs in dataloader:
                        boards, pis, vs = boards.to(device), pis.to(device), vs.to(device)
                        boards = boards.unsqueeze(1); log_probs, value = agent(boards)
                        loss = -torch.sum(pis * log_probs, dim=1).mean() + F.mse_loss(value, vs)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    stage_examples = []

            overall_win_history.extend(stage_wins)
            win_rate = sum(stage_wins) / len(stage_wins) if stage_wins else 0
            print(f"Stage 'Mixed Pool' complete. Win Rate: {win_rate:.2f}")

        else: # Original, non-staged training logic
            print("\n--- Starting MCTS Training ---")
            opponent = get_opponent(args.opponent_type)
            print(f"Training against: {args.opponent_type}")
            all_examples = []
            for i in trange(1, args.n_episodes + 1, desc="MCTS Training"):
                mcts_player = 1 if i % 2 == 0 else -1
                new_examples, winner = execute_episode_vs_opponent(agent, mcts, opponent, mcts_player, args)
                all_examples.extend(new_examples)

                if winner == mcts_player: overall_win_history.append(1)
                else: overall_win_history.append(0)

                if i % args.mcts_epochs == 0 and all_examples:
                    agent.train()
                    dataset = torch.utils.data.TensorDataset(
                        torch.FloatTensor(np.array([e[0] for e in all_examples])),
                        torch.FloatTensor(np.array([e[2] for e in all_examples])),
                        torch.FloatTensor(np.array([e[3] for e in all_examples])).unsqueeze(1)
                    )
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.mcts_batch_size, shuffle=True)
                    for boards, pis, vs in dataloader:
                        boards, pis, vs = boards.to(device), pis.to(device), vs.to(device)
                        boards = boards.unsqueeze(1); log_probs, value = agent(boards)
                        loss = -torch.sum(pis * log_probs, dim=1).mean() + F.mse_loss(value, vs)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    all_examples = []
                
                if i % (args.mcts_epochs * 10) == 0:
                     recent_wins = overall_win_history[-100:]
                     win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                     if overall_loss_history: # Check if there is any loss recorded
                        print(f"\nEpisode {i}: Win Rate (last 100) = {win_rate:.2f}, Avg Loss = {overall_loss_history[-1]:.4f}")
                     else:
                        print(f"\nEpisode {i}: Win Rate (last 100) = {win_rate:.2f}, Avg Loss = N/A")

                     if args.win_rate_threshold and win_rate >= args.win_rate_threshold:
                        print(f"Win rate threshold of {args.win_rate_threshold:.2f} reached. Stopping training.")
                        break
        
        print("--- Training Complete ---")

        if args.output_file:
            torch.save(agent.state_dict(), args.output_file)
            print(f"Model saved to {args.output_file}")
        
        generate_mcts_plots(
            overall_loss_history, 
            overall_win_history,
            "mcts_training_progress.png"
        )

    # --- Play/Test Modes ---
    elif args.mode in ['play', 'test']:
        if not args.model_path or not os.path.exists(args.model_path):
            print("Error: Must provide a valid model path for play/test mode.")
            return
            
        agent.eval()
        print(f"\nModel loaded from {args.model_path}")
        
        if args.mode == 'play':
            # Implement play logic here if needed
            print("Play mode is not fully implemented yet.")
        elif args.mode == 'test':
            # Implement test logic here if needed
            print("Test mode is not fully implemented yet.")

def get_agent_move(agent, board, player, device, board_size, mcts_simulations):
    temp_env = hexPosition(board_size)
    temp_env.board = board
    temp_env.player = player
    action_probs = mcts.get_action_probs(temp_env, mcts_simulations, temp=0)
    best_action = max(action_probs, key=action_probs.get)
    return best_action

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCTS Hex Agent')
    parser.add_argument('--board-size', type=int, default=3, help='Size of the Hex board')
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of episodes for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mcts-simulations', type=int, default=15, help='Number of MCTS simulations per move')
    parser.add_argument('--mcts-epochs', type=int, default=5, help='Number of training epochs per iteration')
    parser.add_argument('--mcts-batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test-episodes', type=int, default=50, help='Number of episodes for post-training test vs random')
    parser.add_argument('--environment', type=str, default='apple', choices=['windows', 'apple', 'kaggle'], help='The training environment to set the device')
    parser.add_argument('--network', type=str, default='cnn', choices=['cnn', 'resnet', 'miniresnet'], help='Network type')
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--save-checkpoints', action='store_true', help='Save model checkpoints after each epoch')
    parser.add_argument('--p1-win-rate-target', type=float, default=0.5, help='Target P1 win rate for dynamic oversampling')
    parser.add_argument('--dynamic-oversampling-strength', type=float, default=0.1, help='Strength of dynamic oversampling')
    parser.add_argument('--model-path', type=str, default=None, help='Path to save the final model.')

    # Staged training arguments
    parser.add_argument('--staged-training', action='store_true', help='Enable staged training against baseline agents.')
    parser.add_argument('--stage1-episodes', type=int, default=20, help='Number of episodes against greedy agent.')
    parser.add_argument('--stage2-episodes', type=int, default=20, help='Number of episodes against aggressive agent.')
    parser.add_argument('--stage3-episodes', type=int, default=20, help='Number of episodes against defensive agent.')
    parser.add_argument('--mixed-pool-episodes', type=int, default=40, help='Number of episodes against a mixed pool of agents.')

    # New arguments for dynamic starting logic
    parser.add_argument('--mcts-dynamic-starts', action='store_true', help='Enable dynamic starting logic.')
    parser.add_argument('--win-rate-threshold', type=float, default=0.75, help='Win rate threshold for stopping training.')
    parser.add_argument('--opponent-type', type=str, default='GreedyAgent', choices=['GreedyAgent', 'RandomAgent'], help='Type of opponent for training.')
    parser.add_argument('--output-file', type=str, default=None, help='Path to save the final trained model.')

    args = parser.parse_args()

    print(f"--- MCTS Hex Agent (Direct Execution) --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    run_mcts(args) 