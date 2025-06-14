#This script uses some packages from the python standard library:
#copy
#random
#pickle
#Of the above only 'copy' is necessary for basic functionality

from datetime import datetime

class hexPosition (object):
    """
    Objects of this class correspond to a game of Hex.
    
    Attributes
    ----------
    size : int 
        The size of the board. The board is 'size*size'.
    board : list[list[int]]
        An array representing the hex board. '0' means empty. '1' means 'white'. '-1' means 'black'.
    player : int
        The player who is currently required to make a move. '1' means 'white'. '-1' means 'black'.
    winner : int
        Whether the game is won and by whom. '0' means 'no winner'. '1' means 'white' has won. '-1' means 'black' has won.
    history : list[list[list[int]]]
        A list of board-state arrays. Stores the history of play.    
    """
    def __init__ (self, size=7):
        #enforce lower and upper bound on size
        size = max(2,min(size,26))
        #attributes encoding a game state
        self.size = max(2,min(size,26))
        board = [[0 for x in range(size)] for y in range(size)]
        self.board = board
        self.player = 1
        self.winner = 0
        #attributes storing the history
        self.history = [board]
    def reset (self):
        """
        This method resets the hex board. All stones are removed from the board and the history is cleared.
        """
        self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        self.player = 1
        self.winner = 0
        self.history = []
    def move (self, coordinates):
        """
        This method enacts a move.
        The variable 'coordinates' is a tuple of board coordinates.
        The variable 'player_num' is either 1 (white) or -1 (black).
        """
        assert (self.winner == 0), "The game is already won."
        assert (self.board[coordinates[0]][coordinates[1]] == 0), "These coordinates already contain a stone."
        from copy import deepcopy
        #make the move
        self.board[coordinates[0]][coordinates[1]] = self.player
        #change the active player
        self.player *= -1
        #evaluate the position
        self.evaluate()
        #append to history
        self.history.append(deepcopy(self.board))
    def print (self, invert_colors=True):
        """
        This method prints a visualization of the hex board to the standard output.
        If the standard output prints black text on a white background, one must set invert_colors=False.
        """
        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " "*5+(" "*3).join(names[:self.size])
        print(headings)
        tops = " "*5+(" "*3).join("_"*self.size)
        print(tops)
        roof = " "*4+"/ \\"+"_/ \\"*(self.size-1)
        print(roof)
        #color mapping inverted by default for display in terminal.
        if invert_colors:
            color_mapping = lambda i: " " if i==0 else ("\u25CB" if i== -1 else "\u25CF")
        else:
            color_mapping = lambda i: " " if i==0 else ("\u25CF" if i== -1 else "\u25CB")
        for r in range(self.size):
            row_mid = " "*indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping,self.board[r]))
            row_mid += " | {} ".format(r+1)
            print(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*self.size
            if r<self.size-1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        print(headings)
    def _get_adjacent (self, coordinates):
        """
        Helper function to obtain adjacent cells in the board array.
        Used in position evaluation to construct paths through the board.
        """
        u = (coordinates[0]-1, coordinates[1])
        d = (coordinates[0]+1, coordinates[1])
        r = (coordinates[0], coordinates[1]-1)
        l = (coordinates[0], coordinates[1]+1)
        ur = (coordinates[0]-1, coordinates[1]+1)
        dl = (coordinates[0]+1, coordinates[1]-1)
        return [pair for pair in [u,d,r,l,ur,dl] if max(pair[0], pair[1]) <= self.size-1 and min(pair[0], pair[1]) >= 0]
    def get_action_space (self, recode_black_as_white=False):
        """
        This method returns a list of array positions which are empty (on which stones may be put).
        """
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    actions.append((i,j))
        if recode_black_as_white:
            return [self.recode_coordinates(action) for action in actions]
        else:
            return(actions)
    def _random_move (self):
        """
        This method enacts a uniformly randomized valid move.
        """
        from random import choice
        chosen = choice(self.get_action_space())
        self.move(chosen)
    def _random_match (self):
        """
        This method randomizes an entire playthrough. Mostly useful to test code functionality.
        """
        while self.winner == 0:
            self._random_move()
    def _prolong_path (self, path):
        """
        A helper function used for board evaluation.
        """
        player = self.board[path[-1][0]][path[-1][1]]
        candidates = self._get_adjacent(path[-1])
        #preclude loops
        candidates = [cand for cand in candidates if cand not in path]
        candidates = [cand for cand in candidates if self.board[cand[0]][cand[1]] == player]
        return [path+[cand] for cand in candidates]
    def evaluate (self, verbose=False):
        """
        Evaluates the board position and adjusts the 'winner' attribute of the object accordingly.
        """
        self._evaluate_white(verbose=verbose)
        self._evaluate_black(verbose=verbose)  
    def _evaluate_white (self, verbose):
        """
        Evaluate whether the board position is a win for player '1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.board[i][0] == 1:
                paths.append([(i,0)])
                visited.append([(i,0)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][1] == self.size-1:
                        if verbose:
                            print("A winning path for 'white' ('1'):\n",new)
                        self.winner = 1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])
    def _evaluate_black (self, verbose):
        """
        Evaluate whether the board position is a win for player '-1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.board[0][i] == -1:
                paths.append([(0,i)])
                visited.append([(0,i)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][0] == self.size-1:
                        if verbose:
                            print("A winning path for 'black' ('-1'):\n",new)
                        self.winner = -1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])
    def human_vs_machine (self, human_player=1, machine=None):
        """
        Allows a human to play against a machine agent.
        """
        from plotting import plot_board_state # Lazy import for plotting
        
        while self.winner == 0:
            self.print()
            if self.player == human_player:
                try:
                    move_str = input(f"Player {self.player}, enter your move (e.g., 'A1') or 's' to save board: ")
                    if move_str.lower() == 's':
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"board_snapshot_{timestamp}.png"
                        plot_board_state(self.board, filename)
                        continue

                    col = ord(move_str[0].upper()) - ord('A')
                    row = int(move_str[1:]) - 1
                    
                    if self.board[row][col] != 0:
                        print("Invalid move: cell is already occupied.")
                        continue
                    self.move((row, col))
                except (ValueError, IndexError):
                    print("Invalid input format. Please use format like 'A1'.")
            else:
                print(f"Machine player {self.player} is thinking...")
                move = machine(self.board, self.get_action_space())
                self.move(move)
                print(f"Machine chose move: {chr(ord('A') + move[1])}{move[0] + 1}")
        
        self.print()
        print(f"\n--- Game Over ---")
        print(f"Winner is Player {self.winner}")
    def recode_black_as_white (self, print=False, invert_colors=True):
        """
        Returns a board where black is recoded as white and wants to connect horizontally.
        This corresponds to flipping the board along the south-west to north-east diagonal and swapping colors.
        This may be used to train AI players in a 'color-blind' way.
        """
        flipped_board = [[0 for i in range(self.size)] for j in range(self.size)]
        #flipping and color change
        for i in range(self.size):
            for j in range(self.size):
                if self.board[self.size-1-j][self.size-1-i] == 1:
                    flipped_board[i][j] = -1
                if self.board[self.size-1-j][self.size-1-i] == -1:
                    flipped_board[i][j] = 1
        return flipped_board
    def recode_coordinates (self, coordinates):
        """
        Transforms a coordinate tuple (with respect to the board) analogously to the method recode_black_as_white.
        """
        assert(0 <= coordinates[0] and self.size-1 >= coordinates[0]), "There is something wrong with the first coordinate."
        assert(0 <= coordinates[1] and self.size-1 >= coordinates[1]), "There is something wrong with the second coordinate."
        return (self.size-1-coordinates[1], self.size-1-coordinates[0])
    def coordinate_to_scalar (self, coordinates):
        """
        Helper function to convert coordinates to scalars.
        This may be used as alternative coding for the action space.
        """
        assert(0 <= coordinates[0] and self.size-1 >= coordinates[0]), "There is something wrong with the first coordinate."
        assert(0 <= coordinates[1] and self.size-1 >= coordinates[1]), "There is something wrong with the second coordinate."
        return coordinates[0]*self.size + coordinates[1]
    def scalar_to_coordinates (self, scalar):
        """
        Helper function to transform a scalar "back" to coordinates.
        Reverses the output of 'coordinate_to_scalar'.
        """
        coord1 = int(scalar/self.size)
        coord2 = scalar - coord1 * self.size
        assert(0 <= coord1 and self.size-1 >= coord1), "The scalar input is invalid."
        assert(0 <= coord2 and self.size-1 >= coord2), "The scalar input is invalid."
        return (coord1, coord2)
    def machine_vs_machine (self, machine1=None, machine2=None):
        """
        Let two AIs play a game against each other.
        The variables machine1 and machine2 must point to a function that maps a board state and an action set to an element of the action set.
        If a machine is not specified random actions will be used.
        This method should not be used for training an algorithm.
        """
        #use random players as default if machines not specified
        if machine1 == None:
            def machine1 (board, action_list):
                from random import choice
                return choice(action_list)
        if machine2 == None:
            def machine2 (board, action_list):
                from random import choice
                return choice(action_list)
        #the match
        self.reset()
        while self.winner == 0:
            self.print()
            input("Press ENTER to continue.")
            if self.player == 1:
                chosen = machine1(self.board, self.get_action_space())
            if self.player == -1:
                chosen = machine2(self.board, self.get_action_space())
            self.move(chosen)
            if self.winner == 1:
                self.print()
                self._evaluate_white(verbose=True)
            if self.winner == -1:
                self.print()
                self._evaluate_black(verbose=True)
    def machine_vs_machine_silent(self, machine1=None, machine2=None):
        """
        Simulates a game between two machine agents without printing to the console.
        Returns the winner.
        """
        self.reset()
        
        while self.winner == 0:
            possible_actions = self.get_action_space()
            if not possible_actions:
                break # Draw or game over
                
            if self.player == 1:
                chosen = machine1(self.board, possible_actions, self.player)
            else: # player is -1
                chosen = machine2(self.board, possible_actions, self.player)
            
            self.move(chosen)
            
        return self.winner
    def replay_history (self):
        """
        Print the game history to standard output.
        """
        for board in self.history:
            temp = hexPosition(size=self.size)
            temp.board = board
            temp.print()
            input("Press ENTER to continue.")
    def save (self, path):
        """
        Serializes the object as a bytestream.
        """
        import pickle
        file = open(path, 'ab')
        pickle.dump(self, file)                     
        file.close()
    def clone(self):
        """
        Creates a deep copy of the game state, but much faster than
        the generic deepcopy, as we only copy what's necessary.
        """
        new_pos = hexPosition(self.size)
        new_pos.board = [row[:] for row in self.board]
        new_pos.player = self.player
        new_pos.winner = self.winner
        # We don't need to copy the history for MCTS simulations
        new_pos.history = []
        return new_pos
