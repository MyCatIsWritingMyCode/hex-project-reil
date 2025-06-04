import hex_engine as engine
import numpy as np
import matplotlib.pyplot as plt
from submission.facade import agent
from tqdm import tqdm

def run_test_games(num_games=100):
    """
    Run test games between our agent and random moves.
    
    Args:
        num_games: Number of games to play
        
    Returns:
        tuple: (wins, losses, draws, game_lengths)
    """
    wins = 0
    losses = 0
    draws = 0
    game_lengths = []
    
    for _ in tqdm(range(num_games), desc="Playing games"):
        # Initialize new game
        game = engine.hexPosition()
        
        # Play until game is over
        while game.winner == 0:
            # Our agent plays as white (1)
            if game.player == 1:
                action_set = game.get_action_space()
                move = agent(game.board, action_set)
                game.move(move)
            # Random moves for black (-1)
            else:
                game._random_move()
                
        # Record results
        if game.winner == 1:
            wins += 1
        elif game.winner == -1:
            losses += 1
        else:
            draws += 1
            
        game_lengths.append(len(game.history))
        
    return wins, losses, draws, game_lengths

def plot_results(wins, losses, draws, game_lengths):
    """
    Plot the results of our test games.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot win/loss/draw distribution
    labels = ['Wins', 'Losses', 'Draws']
    values = [wins, losses, draws]
    colors = ['green', 'red', 'gray']
    
    ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('Game Outcomes')
    
    # Plot game length distribution
    ax2.hist(game_lengths, bins=20, color='blue', alpha=0.7)
    ax2.set_title('Game Length Distribution')
    ax2.set_xlabel('Number of Moves')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def main():
    print("Testing greedy agent against random moves...")
    wins, losses, draws, game_lengths = run_test_games(num_games=100)
    
    # Print results
    total_games = wins + losses + draws
    print(f"\nResults after {total_games} games:")
    print(f"Wins: {wins} ({wins/total_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/total_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
    print(f"\nAverage game length: {np.mean(game_lengths):.1f} moves")
    print(f"Shortest game: {min(game_lengths)} moves")
    print(f"Longest game: {max(game_lengths)} moves")
    
    # Plot results
    plot_results(wins, losses, draws, game_lengths)
    print("\nResults have been saved to 'test_results.png'")

if __name__ == "__main__":
    main() 