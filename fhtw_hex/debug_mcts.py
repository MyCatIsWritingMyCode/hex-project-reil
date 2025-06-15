#!/usr/bin/env python3
"""
Debug script to test MCTS agent behavior and identify issues.
"""

import torch
import numpy as np
from hex_engine import hexPosition
from mcts_agent import MCTS
from networks import MiniResNet
from baseline_agents import RandomAgent

def test_mcts_basic_functionality():
    """Test basic MCTS functionality."""
    print("=== MCTS Basic Functionality Test ===")
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    board_size = 7
    action_space_size = board_size ** 2
    
    # Create network and MCTS
    agent = MiniResNet(board_size, action_space_size).to(device)
    mcts = MCTS(agent, device=device)
    
    # Test on empty board
    env = hexPosition(board_size)
    print(f"Empty board, player: {env.player}")
    
    # Test network output
    board_tensor = torch.FloatTensor(env.board).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        policy, value = agent(board_tensor)
        policy = torch.exp(policy)  # Convert log_probs to probs
    
    print(f"Network value: {value.item():.4f}")
    print(f"Policy sum: {policy.sum().item():.4f}")
    print(f"Policy max: {policy.max().item():.4f}")
    print(f"Policy min: {policy.min().item():.4f}")
    
    # Test MCTS search
    print("\n--- Testing MCTS Search ---")
    action_probs = mcts.get_action_probs(env, num_simulations=10)
    print(f"MCTS returned {len(action_probs)} actions")
    print(f"Action probs sum: {sum(action_probs.values()):.4f}")
    
    if action_probs:
        best_action = max(action_probs, key=action_probs.get)
        print(f"Best action: {best_action}, prob: {action_probs[best_action]:.4f}")
    
    return action_probs

def test_game_simulation():
    """Test a full game simulation."""
    print("\n=== Game Simulation Test ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    board_size = 7
    action_space_size = board_size ** 2
    
    # Create agents
    agent = MiniResNet(board_size, action_space_size).to(device)
    mcts = MCTS(agent, device=device)
    random_agent = RandomAgent()
    
    # Play a game
    env = hexPosition(board_size)
    move_count = 0
    
    while env.winner == 0 and move_count < 20:  # Limit moves for debugging
        current_player = env.player
        print(f"\nMove {move_count + 1}, Player {current_player}")
        
        if current_player == 1:  # MCTS plays as player 1
            action_probs = mcts.get_action_probs(env, num_simulations=5)  # Few sims for speed
            if action_probs:
                action = max(action_probs, key=action_probs.get)
                print(f"MCTS chooses: {action}")
            else:
                print("MCTS returned no actions!")
                break
        else:  # Random agent plays as player -1
            valid_actions = env.get_action_space()
            action = random_agent.select_move(env.board, valid_actions, current_player)
            print(f"Random chooses: {action}")
        
        env.move(action)
        move_count += 1
        
        # Print board state
        print("Board state:")
        for row in env.board:
            print([f"{x:2d}" for x in row])
    
    print(f"\nGame ended: Winner = {env.winner}, Moves = {move_count}")
    return env.winner

def test_value_consistency():
    """Test if network values make sense."""
    print("\n=== Value Consistency Test ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    board_size = 7
    action_space_size = board_size ** 2
    
    agent = MiniResNet(board_size, action_space_size).to(device)
    
    # Test different board states
    test_cases = [
        ("Empty board", np.zeros((board_size, board_size))),
        ("Player 1 advantage", np.array([[1, 1, 1, 0, 0, 0, 0]] + [[0]*7 for _ in range(6)])),
        ("Player -1 advantage (flipped)", np.array([[0]*7 for _ in range(6)] + [[1, 1, 1, 0, 0, 0, 0]])),  # Flipped to canonical form
        ("Player 1 winning", np.array([[1, 1, 1, 1, 1, 1, 1]] + [[0]*7 for _ in range(6)])),
    ]
    
    for name, board in test_cases:
        board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, value = agent(board_tensor)
        
        print(f"{name}: Value = {value.item():.4f}")

def test_trained_model():
    """Test the trained model values."""
    print("\n=== Trained Model Test ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    board_size = 7
    action_space_size = board_size ** 2
    
    # Load trained model
    agent = MiniResNet(board_size, action_space_size).to(device)
    try:
        agent.load_state_dict(torch.load("mcts_longer_training.pth", map_location=device))
        print("Loaded trained model: mcts_longer_training.pth")
    except:
        print("Could not load trained model, using random weights")
    
    # Test different board states
    test_cases = [
        ("Empty board", np.zeros((board_size, board_size))),
        ("Player 1 advantage", np.array([[1, 1, 1, 0, 0, 0, 0]] + [[0]*7 for _ in range(6)])),
        ("Player -1 advantage (flipped)", np.array([[0]*7 for _ in range(6)] + [[1, 1, 1, 0, 0, 0, 0]])),  # Flipped to canonical form
        ("Player 1 winning", np.array([[1, 1, 1, 1, 1, 1, 1]] + [[0]*7 for _ in range(6)])),
    ]
    
    for name, board in test_cases:
        board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, value = agent(board_tensor)
        
        print(f"{name}: Value = {value.item():.4f}")
    
    return agent

def test_board_perspective():
    """Test how the network sees different board perspectives."""
    print("\n=== Board Perspective Test ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    board_size = 7
    action_space_size = board_size ** 2
    
    # Load trained model
    agent = MiniResNet(board_size, action_space_size).to(device)
    try:
        agent.load_state_dict(torch.load("mcts_final_fix.pth", map_location=device))
        print("Loaded trained model: mcts_final_fix.pth")
    except:
        print("Could not load trained model, using random weights")
    
    # Test the same position from both player perspectives
    # Player -1 has advantage on bottom row
    original_board = np.array([[0]*7 for _ in range(6)] + [[-1, -1, -1, 0, 0, 0, 0]])
    
    print("Original board (Player -1 advantage):")
    for row in original_board:
        print([f"{x:2d}" for x in row])
    
    # Test from Player 1's perspective (no flipping)
    board_tensor_p1 = torch.FloatTensor(original_board).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_p1, value_p1 = agent(board_tensor_p1)
    print(f"From Player 1's perspective: Value = {value_p1.item():.4f}")
    
    # Test from Player -1's perspective (with flipping)
    flipped_board = original_board * -1
    board_tensor_p2 = torch.FloatTensor(flipped_board).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_p2, value_p2 = agent(board_tensor_p2)
    print(f"From Player -1's perspective (flipped): Value = {value_p2.item():.4f}")
    
    print("Flipped board (what Player -1 sees):")
    for row in flipped_board:
        print([f"{x:2d}" for x in row])
    
    return value_p1.item(), value_p2.item()

if __name__ == "__main__":
    print("Starting MCTS Debug Tests...")
    
    # Test 1: Basic functionality
    action_probs = test_mcts_basic_functionality()
    
    # Test 2: Value consistency
    test_value_consistency()
    
    # Test 3: Game simulation
    winner = test_game_simulation()
    
    # Test 4: Trained model
    test_trained_model()
    
    # Test 5: Board perspective
    test_board_perspective()
    
    print("\n=== Debug Summary ===")
    print(f"MCTS returned actions: {'Yes' if action_probs else 'No'}")
    print(f"Game completed: {'Yes' if winner != 0 else 'No'}")
    print("Debug tests completed.") 