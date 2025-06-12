import torch
import torch.nn.functional as F
import numpy as np
import argparse
from hex_engine import hexPosition
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime
import math
from copy import deepcopy
from networks import ActorCritic, ResNet
import pandas as pd
from plotting import generate_training_plots
from torch.distributions import Categorical

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
            while node is not None:
                node.N += 1
                # The value is from the perspective of the player AT THE NODE.
                # Since we flip the board for the network, the value is for the current player.
                # We need to negate the value for the parent node if the parent is the other player.
                node.Q += value
                value *= -1 # The parent's value is the negation of the child's
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
        q_value = child.Q / (child.N + 1e-9)
        # The value is from the perspective of the node's player. 
        # Since we alternate players, we need to flip the sign for the parent's decision.
        # But our Q update already handles this.
        
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
    examples = [(x[0], x[1], x[2], winner * x[1]) for x in train_examples]
    return examples, winner

def run_mcts(args):
    """The main entry point for running the MCTS agent."""
    
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
        
    print(f"Using device: {device}")

    action_space_size = args.board_size**2
    if args.network == 'cnn':
        agent = ActorCritic(args.board_size, action_space_size).to(device)
    elif args.network == 'resnet':
        agent = ResNet(args.board_size, action_space_size).to(device)
    else:
        raise ValueError(f"Unknown network type: {args.network}")
    
    mcts = MCTS(agent, device=device)

    if args.mode == 'play' or args.mode == 'test':
        try:
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            agent.eval()
            print(f"\nModel loaded from {args.model_path}")

            def mcts_player_func(board, _action_set):
                temp_env = hexPosition(args.board_size)
                temp_env.board = board
                num_white_stones = sum(row.count(1) for row in board)
                num_black_stones = sum(row.count(-1) for row in board)
                temp_env.player = 1 if num_white_stones == num_black_stones else -1
                action_probs = mcts.get_action_probs(temp_env, args.mcts_simulations, temp=0)
                best_action = max(action_probs, key=action_probs.get)
                return best_action

            if args.mode == 'play':
                 env = hexPosition(args.board_size)
                 env.human_vs_machine(human_player=args.human_player, machine=mcts_player_func)
            
            if args.mode == 'test':
                print(f"--- Running Baseline Test vs. Random Agent ---")
                env = hexPosition(args.board_size)
                p1_wins = 0
                for _ in trange(args.test_episodes, desc="Agent as P1 (White)"):
                    winner = env.machine_vs_machine_silent(machine1=mcts_player_func)
                    if winner == 1:
                        p1_wins += 1
                
                p2_wins = 0
                for _ in trange(args.test_episodes, desc="Agent as P2 (Black)"):
                    winner = env.machine_vs_machine_silent(machine2=mcts_player_func)
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
        
        # --- Set up logging ---
        training_log = []
        all_train_examples = []
        p1_wins = 0

        print("\n--- Starting Self-Play ---")
        for i in trange(args.n_episodes, desc="Self-Play"):
            new_examples, winner = execute_episode(agent, mcts, args)
            if winner == 1:
                p1_wins += 1
            all_train_examples.extend(new_examples)
            
        # --- Dynamic Oversampling Calculation ---
        p1_win_rate = p1_wins / args.n_episodes
        print(f"\n--- Self-Play Complete: P1 Win Rate was {p1_win_rate:.2%} ---")
        
        # Adjust the P2 oversampling ratio based on P1's performance
        p2_oversampling_ratio = 1.0 + args.dynamic_oversampling_strength * (p1_win_rate - args.p1_win_rate_target)
        p2_oversampling_ratio = max(0.1, p2_oversampling_ratio) # Prevent zero or negative ratios
        print(f"Target P1 Win Rate: {args.p1_win_rate_target:.2%}. Dynamic P2 Oversampling Ratio set to: {p2_oversampling_ratio:.2f}")
        # ---

        print("\n--- Starting Training ---")
        agent.train()

        # Separate data for potential oversampling
        p1_examples = [ex for ex in all_train_examples if ex[1] == 1]
        p2_examples = [ex for ex in all_train_examples if ex[1] == -1]

        for epoch in trange(args.mcts_epochs, desc="Epochs"):
            np.random.shuffle(all_train_examples)
            
            for i in range(0, len(all_train_examples), args.mcts_batch_size):
                p2_batch_size = int(args.mcts_batch_size * (p2_oversampling_ratio / (1 + p2_oversampling_ratio)))
                p1_batch_size = args.mcts_batch_size - p2_batch_size

                p1_sample = [p1_examples[i] for i in np.random.randint(len(p1_examples), size=p1_batch_size)] if p1_batch_size > 0 and len(p1_examples) > 0 else []
                p2_sample = [p2_examples[i] for i in np.random.randint(len(p2_examples), size=p2_batch_size)] if p2_batch_size > 0 and len(p2_examples) > 0 else []
                
                if not p1_sample and not p2_sample: continue

                batch = p1_sample + p2_sample
                np.random.shuffle(batch) # Shuffle the combined batch
                
                boards, _, pis, vs = zip(*batch) # The player (index 1) is no longer needed
                
                boards = torch.FloatTensor(np.array(boards)).unsqueeze(1).to(device)
                target_pis = torch.FloatTensor(np.array(pis)).to(device)
                target_vs = torch.FloatTensor(np.array(vs)).to(device)

                out_pi, out_v = agent(boards)
                
                # --- Calculate Entropy ---
                entropy_dist = Categorical(probs=out_pi)
                policy_entropy = entropy_dist.entropy().mean().item()

                loss_pi = -torch.sum(target_pis * F.log_softmax(out_pi, dim=1)) / target_pis.size()[0]
                loss_v = F.mse_loss(out_v.view(-1), target_vs)
                
                total_loss = loss_pi + loss_v

                # --- Log metrics for this training step (once per epoch for simplicity) ---
                if i == 0:
                    training_log.append({
                        'episode': (epoch + 1) * args.n_episodes, # Approximate episode count
                        'p1_win_rate_batch': p1_win_rate,
                        'p2_oversampling_ratio': p2_oversampling_ratio,
                        'actor_loss': loss_pi.item(),
                        'critic_loss': loss_v.item(),
                        'total_loss': total_loss.item(),
                        'policy_entropy': policy_entropy
                    })

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Save a checkpoint after each epoch if requested
            if args.save_checkpoints:
                save_path = f"mcts_checkpoint_epoch{epoch+1}.pth"
                torch.save(agent.state_dict(), save_path)
                print(f"\nSaved checkpoint to {save_path}")

        # --- Save detailed log and generate plots ---
        if training_log:
            log_df = pd.DataFrame(training_log)
            log_df.to_csv('mcts_training_log.csv', index=False)
            print("Detailed training log saved to mcts_training_log.csv")
            generate_training_plots(log_df, 'MCTS', args.p1_win_rate_target, "mcts_training_progress.png")

        # Save the final trained model
        if args.model_path:
            save_path = args.model_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"mcts_agent_{args.network}_{args.board_size}x{args.board_size}_{timestamp}.pth"
        torch.save(agent.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

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
    parser.add_argument('--environment', type=str, default='apple', choices=['windows', 'apple', 'kaggle'], help='The training environment to set the device')
    parser.add_argument('--network', type=str, default='cnn', choices=['cnn', 'resnet'], help='Network type')
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--save-checkpoints', action='store_true', help='Save model checkpoints after each epoch')
    parser.add_argument('--p1-win-rate-target', type=float, default=0.5, help='Target P1 win rate for dynamic oversampling')
    parser.add_argument('--dynamic-oversampling-strength', type=float, default=0.1, help='Strength of dynamic oversampling')

    args = parser.parse_args()

    print(f"--- MCTS Hex Agent (Direct Execution) --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    run_mcts(args) 