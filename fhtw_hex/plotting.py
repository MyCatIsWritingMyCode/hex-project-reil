import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_training_plots(log_df, agent_type, p1_win_rate_target, output_filename="training_progress.png"):
    """
    Generates a comprehensive multi-panel plot from a training log DataFrame.
    
    Args:
        log_df (pd.DataFrame): DataFrame containing the training log data.
        agent_type (str): The type of agent (e.g., 'A2C', 'MCTS').
        p1_win_rate_target (float): The target win rate for P1.
        output_filename (str): The filename to save the plot to.
    """
    if log_df.empty:
        print("Warning: Training log is empty. Skipping plot generation.")
        return
        
    # Determine which plots to generate based on available columns
    has_win_rate_data = 'p1_win_rate_batch' in log_df.columns and 'p2_oversampling_ratio' in log_df.columns
    
    num_plots = 2 + (2 if has_win_rate_data else 0)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
    fig.suptitle(f'{agent_type.upper()} Training Analysis', fontsize=16)

    plot_idx = 0

    if has_win_rate_data:
        # Plot Win Rate
        axs[plot_idx].plot(log_df.get('episode', log_df.index), log_df['p1_win_rate_batch'], label='P1 Win Rate (in Batch)', color='b', marker='.', linestyle='-')
        axs[plot_idx].axhline(y=p1_win_rate_target, color='r', linestyle='--', label=f'Target ({p1_win_rate_target:.0%})')
        axs[plot_idx].set_ylabel('P1 Win Rate')
        axs[plot_idx].set_title('Player 1 Win Rate per Batch')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1

        # Plot Oversampling Ratio
        axs[plot_idx].plot(log_df.get('episode', log_df.index), log_df['p2_oversampling_ratio'], label='P2 Oversampling Ratio', color='g', marker='.', linestyle='-')
        axs[plot_idx].set_ylabel('Ratio')
        axs[plot_idx].set_title('Dynamic Player 2 Oversampling Ratio')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True)
        plot_idx += 1

    # Plot Losses
    axs[plot_idx].plot(log_df.get('epoch', log_df.index), log_df['total_loss'], label='Total Loss', color='r')
    axs[plot_idx].plot(log_df.get('epoch', log_df.index), log_df['actor_loss'], label='Actor/Policy Loss', linestyle='--', alpha=0.7)
    axs[plot_idx].plot(log_df.get('epoch', log_df.index), log_df['critic_loss'], label='Critic/Value Loss', linestyle='--', alpha=0.7)
    axs[plot_idx].set_ylabel('Loss')
    axs[plot_idx].set_title('Training Losses')
    axs[plot_idx].legend()
    axs[plot_idx].grid(True)
    plot_idx += 1
    
    # Plot Policy Entropy
    axs[plot_idx].plot(log_df.get('epoch', log_df.index), log_df['policy_entropy'], label='Policy Entropy', color='purple')
    axs[plot_idx].set_xlabel('Epoch / Episode')
    axs[plot_idx].set_ylabel('Entropy')
    axs[plot_idx].set_title('Policy Entropy (Agent Certainty)')
    axs[plot_idx].legend()
    axs[plot_idx].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_filename)
    print(f"Training analysis plot saved to {output_filename}")

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