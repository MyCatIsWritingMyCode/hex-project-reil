import numpy as np
from typing import List, Tuple

class GreedyAgent:
    def __init__(self, connection_weight=1.0, blocking_weight=0.8, distance_weight=0.5):
        # We'll use these weights to balance different strategic elements
        self.connection_weight = connection_weight
        self.blocking_weight = blocking_weight
        self.distance_weight = distance_weight

    def evaluate_move(self, board: List[List[int]], move: Tuple[int, int], player: int) -> float:
        """
        Evaluate how good a move is based on simple heuristics.
        
        Args:
            board: Current game state
            move: Position to evaluate (row, col)
            player: Current player (1 for white, -1 for black)
            
        Returns:
            float: Score for this move (higher is better)
        """
        row, col = move
        score = 0.0
        
        # Check how many friendly pieces are connected
        connected_pieces = self._count_connected_pieces(board, move, player)
        score += self.connection_weight * connected_pieces
        
        # Check how many opponent paths we're blocking
        blocked_paths = self._count_blocked_paths(board, move, player)
        score += self.blocking_weight * blocked_paths
        
        # Check distance to opponent's path
        distance_score = self._calculate_distance_score(board, move, player)
        score += self.distance_weight * distance_score
        
        return score

    def _count_connected_pieces(self, board: List[List[int]], move: Tuple[int, int], player: int) -> int:
        """
        Count how many friendly pieces are adjacent to the move.
        """
        row, col = move
        count = 0
        size = len(board)
        
        # Check all 6 possible directions in hex grid
        directions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (-1, 1),  # up-right
            (1, -1)   # down-left
        ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < size and 0 <= new_col < size and 
                board[new_row][new_col] == player):
                count += 1
                
        return count

    def _count_blocked_paths(self, board: List[List[int]], move: Tuple[int, int], player: int) -> int:
        """
        Count how many potential opponent paths we're blocking.
        """
        opponent = -player
        row, col = move
        count = 0
        size = len(board)
        
        # Check if move blocks any opponent's potential winning paths
        if player == 1:  # White player (horizontal)
            # Check if we're blocking a potential vertical path
            if 0 < row < size-1:  # Not on the edges
                if (board[row-1][col] == opponent and 
                    board[row+1][col] == opponent):
                    count += 1
        else:  # Black player (vertical)
            # Check if we're blocking a potential horizontal path
            if 0 < col < size-1:  # Not on the edges
                if (board[row][col-1] == opponent and 
                    board[row][col+1] == opponent):
                    count += 1
                    
        return count

    def _calculate_distance_score(self, board: List[List[int]], move: Tuple[int, int], player: int) -> float:
        """
        Calculate a score based on distance to opponent's path.
        """
        row, col = move
        size = len(board)
        
        if player == 1:  # White player (horizontal)
            # Distance to left edge (0) and right edge (size-1)
            return 1.0 / (1 + min(col, size-1-col))
        else:  # Black player (vertical)
            # Distance to top edge (0) and bottom edge (size-1)
            return 1.0 / (1 + min(row, size-1-row))

    def select_move(self, board: List[List[int]], action_set: List[Tuple[int, int]], player: int) -> Tuple[int, int]:
        """
        Select the best move based on our evaluation function.
        
        Args:
            board: Current game state
            action_set: List of valid moves
            player: Current player (1 for white, -1 for black)
            
        Returns:
            Tuple[int, int]: Best move coordinates
        """
        if not action_set:
            # In a real game, this means the board is full. For safety, return a random valid move if any, or raise error.
            if action_set:
                return action_set[np.random.randint(len(action_set))]
            raise ValueError("No valid moves available")
            
        # Evaluate each possible move
        move_scores = [(move, self.evaluate_move(board, move, player)) 
                      for move in action_set]
        
        # Add some randomness to avoid getting stuck in patterns
        # but still prefer better moves
        scores = np.array([score for _, score in move_scores])
        probs = np.exp(scores) / np.sum(np.exp(scores))  # Softmax
        
        # Select move based on probabilities
        selected_idx = np.random.choice(len(move_scores), p=probs)
        return move_scores[selected_idx][0]

class AggressiveAgent(GreedyAgent):
    """An agent that prioritizes moving forward across the board."""
    def __init__(self):
        super().__init__(connection_weight=0.2, blocking_weight=0.5, distance_weight=1.5)

class DefensiveAgent(GreedyAgent):
    """An agent that prioritizes blocking and building strong connections."""
    def __init__(self):
        super().__init__(connection_weight=1.5, blocking_weight=1.2, distance_weight=0.2) 