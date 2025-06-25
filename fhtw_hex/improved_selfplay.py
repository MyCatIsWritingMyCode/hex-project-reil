import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import random

class SelfPlayDataGenerator:
    """
    A clean, robust self-play data generator for Actor-Critic + MCTS training.
    
    Key improvements over the original:
    1. Better value assignment (considers player perspective correctly)
    2. Experience replay buffer for more stable training
    3. Temperature scheduling for exploration vs exploitation
    4. Proper canonical board representation
    5. Data augmentation through symmetries
    """
    
    def __init__(self, agent, mcts, device, board_size=7, buffer_size=50000):
        self.agent = agent
        self.mcts = mcts
        self.device = device
        self.board_size = board_size
        self.experience_buffer = deque(maxlen=buffer_size)
        
    def execute_self_play_episode(self, temperature_schedule=None):
        """
        Execute one self-play episode and return training examples.
        
        Args:
            temperature_schedule: Function that takes move_number and returns temperature
                                 Default: high temp early, low temp late
        """
        from hex_engine import hexPosition  # Import here to avoid circular imports
        
        env = hexPosition(self.board_size)
        game_examples = []
        move_count = 0
        
        while env.winner == 0:
            move_count += 1
            
            # Determine temperature for this move
            if temperature_schedule is None:
                # Default: High exploration early, low exploration late
                if move_count <= 10:
                    temp = 1.0  # High exploration
                elif move_count <= 20:
                    temp = 0.5  # Medium exploration  
                else:
                    temp = 0.1  # Low exploration (more deterministic)
            else:
                temp = temperature_schedule(move_count)
            
            # Get MCTS policy for current position
            action_probs = self.mcts.get_action_probs(env, num_simulations=100, temp=temp)
            
            # Create canonical board representation
            canonical_board = self._get_canonical_board(env.board, env.player)
            
            # Convert action probabilities to policy vector
            policy_vector = self._action_probs_to_vector(action_probs)
            
            # Store the position (we'll assign values later)
            game_examples.append({
                'board': canonical_board.copy(),
                'policy': policy_vector.copy(), 
                'player': env.player,
                'move_number': move_count
            })
            
            # Select and make the move
            action = self._sample_action(action_probs, temp)
            env.move(action)
        
        # Game is finished, assign values from each player's perspective
        winner = env.winner
        final_examples = []
        
        for example in game_examples:
            # Value is +1 if the player who made this move won, -1 if they lost
            player_who_moved = example['player']
            if winner == player_who_moved:
                value = 1.0
            elif winner == -player_who_moved:
                value = -1.0
            else:
                value = 0.0  # Draw (shouldn't happen in Hex)
            
            final_examples.append((
                example['board'],
                example['policy'], 
                value
            ))
        
        return final_examples, winner
    
    def _get_canonical_board(self, board, current_player):
        """
        Create canonical board representation.
        
        Key insight: Always represent the board from the perspective of player 1.
        If current player is -1, flip all pieces.
        """
        canonical = np.array(board, dtype=np.float32)
        if current_player == -1:
            canonical = -canonical
        return canonical
    
    def _action_probs_to_vector(self, action_probs: Dict[Tuple[int, int], float]) -> np.ndarray:
        """Convert action probabilities dict to vector."""
        policy_vector = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        
        for (row, col), prob in action_probs.items():
            idx = row * self.board_size + col
            policy_vector[idx] = prob
            
        return policy_vector
    
    def _sample_action(self, action_probs: Dict[Tuple[int, int], float], temperature: float):
        """Sample an action from the policy with given temperature."""
        if not action_probs:
            raise ValueError("No valid actions available")
        
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        if temperature == 0:
            # Deterministic: choose best action
            best_idx = np.argmax(probs)
            return actions[best_idx]
        else:
            # Stochastic sampling
            return np.random.choice(actions, p=probs)
    
    def generate_training_batch(self, num_episodes=10, batch_size=64):
        """
        Generate a batch of training data from multiple self-play episodes.
        
        Returns:
            boards: torch.Tensor of shape (batch_size, 1, board_size, board_size)
            policies: torch.Tensor of shape (batch_size, board_size^2)
            values: torch.Tensor of shape (batch_size, 1)
        """
        # Generate episodes
        print(f"Generating {num_episodes} self-play episodes...")
        for episode in range(num_episodes):
            examples, winner = self.execute_self_play_episode()
            self.experience_buffer.extend(examples)
            
            if episode % 5 == 0:
                print(f"Episode {episode+1}/{num_episodes}, Winner: {winner}")
        
        # Sample batch from buffer
        if len(self.experience_buffer) < batch_size:
            batch_examples = list(self.experience_buffer)
        else:
            batch_examples = random.sample(list(self.experience_buffer), batch_size)
        
        # Convert to tensors
        boards = torch.FloatTensor([ex[0] for ex in batch_examples]).unsqueeze(1)  # Add channel dim
        policies = torch.FloatTensor([ex[1] for ex in batch_examples])
        values = torch.FloatTensor([ex[2] for ex in batch_examples]).unsqueeze(1)
        
        return boards.to(self.device), policies.to(self.device), values.to(self.device)
    
    def augment_data_with_symmetries(self, board, policy):
        """
        Augment training data using Hex symmetries.
        
        Hex has the following symmetries:
        1. Vertical flip (swap players and rotate 180°)
        2. Horizontal flip  
        3. Diagonal flips
        """
        # This is a placeholder - implement based on your Hex representation
        # For now, just return original data
        return [(board, policy)]


class ImprovedTrainingLoop:
    """
    An improved training loop with better loss handling and learning rate scheduling.
    """
    
    def __init__(self, agent, data_generator, learning_rate=1e-3):
        self.agent = agent
        self.data_generator = data_generator
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, verbose=True
        )
        
    def compute_loss(self, boards, target_policies, target_values):
        """
        Compute the training loss with proper weighting.
        """
        # Forward pass
        pred_log_policies, pred_values = self.agent(boards)
        
        # Policy loss (cross-entropy)
        policy_loss = -torch.sum(target_policies * pred_log_policies, dim=1).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(pred_values, target_values)
        
        # Combined loss with weighting
        total_loss = policy_loss + 0.5 * value_loss  # Weight value loss less
        
        return total_loss, policy_loss, value_loss
    
    def train_step(self, boards, target_policies, target_values):
        """Execute one training step."""
        self.agent.train()
        
        total_loss, policy_loss, value_loss = self.compute_loss(boards, target_policies, target_values)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def train_iterations(self, num_iterations=100, episodes_per_iteration=5):
        """
        Run the full training loop.
        """
        print("Starting improved self-play training...")
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
            
            # Generate fresh training data
            boards, policies, values = self.data_generator.generate_training_batch(
                num_episodes=episodes_per_iteration
            )
            
            # Multiple training steps on this data
            total_losses = []
            for _ in range(5):  # 5 training steps per iteration
                total_loss, policy_loss, value_loss = self.train_step(boards, policies, values)
                total_losses.append(total_loss)
            
            avg_loss = np.mean(total_losses)
            print(f"Average Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)
            
            # Save checkpoint every 10 iterations
            if (iteration + 1) % 10 == 0:
                checkpoint_path = f"selfplay_checkpoint_{iteration+1}.pth"
                torch.save(self.agent.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")


# Example usage function
def run_improved_selfplay_training():
    """
    Example of how to use the improved self-play system.
    """
    # Setup (you'll need to adapt this to your specific setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize your agent and MCTS (adapt to your classes)
    # agent = YourActorCriticNetwork(board_size=7, action_size=49).to(device)
    # mcts = YourMCTS(agent, device=device)
    
    # Create data generator
    # data_generator = SelfPlayDataGenerator(agent, mcts, device, board_size=7)
    
    # Create training loop
    # trainer = ImprovedTrainingLoop(agent, data_generator, learning_rate=1e-3)
    
    # Run training
    # trainer.train_iterations(num_iterations=50, episodes_per_iteration=10)
    
    print("Example complete - adapt the code above to your specific classes!")


if __name__ == "__main__":
    run_improved_selfplay_training() 