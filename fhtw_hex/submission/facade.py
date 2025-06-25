# This is the main interface file for the Hex tournament.

# It imports the agent class from your agent file...
from .a2c_final_agent import FinalA2CAgent

# ...and creates a single instance of it.
_agent = FinalA2CAgent()

def agent(board, action_set):
    """
    Main agent interface that the hex engine will call.
    
    Args:
        board: Current game state
        action_set: List of valid moves
        
    Returns:
        Tuple[int, int]: Selected move coordinates
    """
    # Determine current player from the board state
    white_count = sum(row.count(1) for row in board)
    black_count = sum(row.count(-1) for row in board)
    current_player = 1 if white_count == black_count else -1
    
    return _agent.select_move(board, action_set, current_player) 