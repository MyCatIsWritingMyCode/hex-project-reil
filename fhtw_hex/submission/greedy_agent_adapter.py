from submission.greedy_agent import GreedyAgent

def create_greedy_player():
    """
    Creates a callable function for the tournament from the GreedyAgent class.
    """
    agent = GreedyAgent()
    
    def greedy_player_func(board, action_set):
        # The GreedyAgent's select_move method needs to know the current player.
        # We can determine this by counting the stones on the board.
        num_white_stones = sum(row.count(1) for row in board)
        num_black_stones = sum(row.count(-1) for row in board)
        player = 1 if num_white_stones == num_black_stones else -1
        
        return agent.select_move(board, action_set, player)
        
    return greedy_player_func 