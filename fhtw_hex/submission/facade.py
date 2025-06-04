from .greedy_agent import GreedyAgent

# Create a single instance of our agent
_agent = GreedyAgent()

def agent(board, action_set):
    """
    Main agent interface that the hex engine will call.
    
    Args:
        board: Current game state
        action_set: List of valid moves
        
    Returns:
        Tuple[int, int]: Selected move coordinates
    """
    # Determine current player from the board
    # Count pieces to determine whose turn it is
    white_count = sum(row.count(1) for row in board)
    black_count = sum(row.count(-1) for row in board)
    current_player = 1 if white_count == black_count else -1
    
    return _agent.select_move(board, action_set, current_player)

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.