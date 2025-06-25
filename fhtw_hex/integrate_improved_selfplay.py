"""
Integration example: How to integrate the improved self-play into your existing codebase
"""

import torch
import numpy as np
from improved_selfplay import SelfPlayDataGenerator, ImprovedTrainingLoop

# Import your existing classes
from networks import ActorCritic, ResNet
from mcts_agent import MCTS
from hex_engine import hexPosition

def integrate_with_your_code():
    """
    Shows how to integrate the improved self-play with your existing code.
    """
    print("=== Integration Example ===")
    
    # 1. Setup device and parameters
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    board_size = 7
    action_size = board_size * board_size
    
    print(f"Using device: {device}")
    print(f"Board size: {board_size}x{board_size}")
    
    # 2. Initialize your network (use your existing networks)
    print("\nInitializing network...")
    agent = ActorCritic(board_size, action_size).to(device)  # or ResNet
    
    # Optionally load a pre-trained model
    # agent.load_state_dict(torch.load("your_pretrained_model.pth", map_location=device))
    
    # 3. Initialize MCTS with your existing implementation
    print("Initializing MCTS...")
    mcts = MCTS(agent, c_puct=1.0, device=device)
    
    # 4. Create the improved data generator
    print("Creating improved self-play data generator...")
    data_generator = SelfPlayDataGenerator(
        agent=agent, 
        mcts=mcts, 
        device=device, 
        board_size=board_size,
        buffer_size=10000  # Smaller buffer for testing
    )
    
    # 5. Create the improved training loop
    print("Creating improved training loop...")
    trainer = ImprovedTrainingLoop(
        agent=agent, 
        data_generator=data_generator, 
        learning_rate=1e-3
    )
    
    # 6. Run a few training iterations to test
    print("\nRunning test training...")
    trainer.train_iterations(num_iterations=3, episodes_per_iteration=2)
    
    # 7. Save the trained model
    output_path = "improved_selfplay_test.pth"
    torch.save(agent.state_dict(), output_path)
    print(f"\nModel saved to: {output_path}")

def fix_your_current_problems():
    """
    Identifies and fixes the main problems in your current self-play implementation.
    """
    print("\n=== Probleme in eurem aktuellen Code ===")
    
    print("1. VALUE ASSIGNMENT Problem:")
    print("   ❌ Ihr setzt alle Values auf +1 oder -1 basierend auf winner")
    print("   ✅ Sollte sein: value = +1 wenn der Spieler der den Zug machte gewonnen hat")
    print()
    
    print("2. CANONICAL BOARD Problem:")
    print("   ❌ Board representation ist inconsistent zwischen Spielern")
    print("   ✅ Immer vom Spieler 1 Perspektive darstellen")
    print()
    
    print("3. TEMPERATURE Problem:")
    print("   ❌ Konstante temperature während des ganzen Spiels")
    print("   ✅ Hohe temp am Anfang, niedrige temp am Ende")
    print()
    
    print("4. MEMORY MANAGEMENT Problem:")
    print("   ❌ Alle examples gleichzeitig im memory")
    print("   ✅ Experience replay buffer mit max size")
    print()
    
    print("5. LOSS FUNCTION Problem:")
    print("   ❌ Keine Gewichtung zwischen policy und value loss")
    print("   ✅ Value loss sollte weniger gewichtet werden")

def demonstrate_key_fixes():
    """
    Shows the key fixes compared to your original code.
    """
    print("\n=== Key Fixes Demonstrated ===")
    
    # Fix 1: Better value assignment
    print("1. Better Value Assignment:")
    print("   Original code:")
    print("   if winner == 1: value = 1.0")
    print("   elif winner == -1: value = -1.0")
    print()
    print("   Fixed code:")
    print("   player_who_moved = example['player']")
    print("   if winner == player_who_moved: value = 1.0")
    print("   elif winner == -player_who_moved: value = -1.0")
    print()
    
    # Fix 2: Temperature scheduling
    print("2. Temperature Scheduling:")
    print("   Original: temp=1.0 (konstant)")
    print("   Fixed: temp=1.0 früh → temp=0.1 spät")
    print()
    
    # Fix 3: Experience replay
    print("3. Experience Replay Buffer:")
    print("   Original: alle examples auf einmal trainieren")
    print("   Fixed: buffer mit max size, sampling")
    print()
    
    # Fix 4: Loss weighting
    print("4. Loss Function Weighting:")
    print("   Original: policy_loss + value_loss")
    print("   Fixed: policy_loss + 0.5 * value_loss")

def quick_comparison_test():
    """
    Quick test to compare old vs new approach.
    """
    print("\n=== Quick Comparison Test ===")
    
    # Simulate some game data
    winner = 1
    game_moves = [
        {'player': 1, 'move': 1},
        {'player': -1, 'move': 2}, 
        {'player': 1, 'move': 3},
        {'player': -1, 'move': 4},
        {'player': 1, 'move': 5}  # Player 1 wins
    ]
    
    print("Game sequence:")
    for i, move in enumerate(game_moves):
        print(f"  Move {i+1}: Player {move['player']}")
    print(f"Winner: Player {winner}")
    print()
    
    print("Value assignment comparison:")
    print("Method       | Move 1 | Move 2 | Move 3 | Move 4 | Move 5")
    print("-------------|--------|--------|--------|--------|--------")
    
    # Your original method (problematic)
    old_values = [1.0 if winner == 1 else -1.0 for _ in game_moves]
    print(f"Original     | {' | '.join([f'{v:5.1f}' for v in old_values])}")
    
    # Improved method
    new_values = []
    for move in game_moves:
        if winner == move['player']:
            new_values.append(1.0)
        else:
            new_values.append(-1.0)
    print(f"Improved     | {' | '.join([f'{v:5.1f}' for v in new_values])}")
    
    print("\nErklärung:")
    print("- Move 1,3,5: Player 1 macht Zug, Player 1 gewinnt → +1.0")
    print("- Move 2,4: Player -1 macht Zug, Player 1 gewinnt → -1.0")

if __name__ == "__main__":
    # Run all demonstrations
    fix_your_current_problems()
    demonstrate_key_fixes()
    quick_comparison_test()
    
    print("\n" + "="*50)
    print("Ready to integrate? Run: integrate_with_your_code()")
    print("="*50)
    
    # Uncomment to actually run the integration
    # integrate_with_your_code() 