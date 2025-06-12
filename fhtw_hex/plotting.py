import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_training_plots(log_df, agent_type, p1_win_rate_target):
    """
    Generates a comprehensive multi-panel plot from a training log DataFrame.
    
    Args:
        log_df (pd.DataFrame): DataFrame containing the training log data.
        agent_type (str): The type of agent (e.g., 'A2C', 'MCTS').
        p1_win_rate_target (float): The target win rate for P1.
    """
    if log_df.empty:
        print("Warning: Training log is empty. Skipping plot generation.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)
    fig.suptitle(f'{agent_type.upper()} Training Analysis', fontsize=16)

    # Plot Win Rate
    axs[0].plot(log_df['episode'], log_df['p1_win_rate_batch'], label='P1 Win Rate (in Batch)', color='b', marker='.', linestyle='-')
    axs[0].axhline(y=p1_win_rate_target, color='r', linestyle='--', label=f'Target ({p1_win_rate_target:.0%})')
    axs[0].set_ylabel('P1 Win Rate')
    axs[0].set_title('Player 1 Win Rate per Batch')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Oversampling Ratio
    axs[1].plot(log_df['episode'], log_df['p2_oversampling_ratio'], label='P2 Oversampling Ratio', color='g', marker='.', linestyle='-')
    axs[1].set_ylabel('Ratio')
    axs[1].set_title('Dynamic Player 2 Oversampling Ratio')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Losses
    axs[2].plot(log_df['episode'], log_df['total_loss'], label='Total Loss', color='r')
    axs[2].plot(log_df['episode'], log_df['actor_loss'], label='Actor/Policy Loss', linestyle='--', alpha=0.7)
    axs[2].plot(log_df['episode'], log_df['critic_loss'], label='Critic/Value Loss', linestyle='--', alpha=0.7)
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Training Losses')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot Policy Entropy
    axs[3].plot(log_df['episode'], log_df['policy_entropy'], label='Policy Entropy', color='purple')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Entropy')
    axs[3].set_title('Policy Entropy (Agent Certainty)')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'{agent_type.lower()}_training_progress.png')
    print(f"Training analysis plot saved to {agent_type.lower()}_training_progress.png")

def plot_board_state(board, filename="hex_board.png"):
    """Saves a visual representation of a Hex board state to a file."""
    size = len(board)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the hex grid
    for r in range(size):
        for c in range(size):
            x = c + 0.5 * r
            y = -r * 0.866
            hexagon = plt.Polygon([
                [x + 0.5, y], [x + 0.25, y - 0.433], [x - 0.25, y - 0.433],
                [x - 0.5, y], [x - 0.25, y + 0.433], [x + 0.25, y + 0.433]
            ], edgecolor='k', facecolor='lightgray')
            ax.add_patch(hexagon)
            
            # Place stones
            if board[r][c] == 1: # Player 1 (White)
                circle = plt.Circle((x, y), 0.3, color='white', ec='black')
                ax.add_patch(circle)
            elif board[r][c] == -1: # Player 2 (Black)
                circle = plt.Circle((x, y), 0.3, color='black', ec='white')
                ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Board state saved to {filename}") 