from typing import List, Tuple

class HexapawnState:
    def __init__(self, board: List[List[str]], to_move="B"):
        self.board = board
        self.to_move = to_move
        self.white_pawns = set()
        self.black_pawns = set()
        for r in range(3):
            for c in range(3):
                if board[r][c] == "w":
                    self.white_pawns.add((r, c))
                elif board[r][c] == "b":
                    self.black_pawns.add((r, c))
    
    @staticmethod
    def to_move(state):
        return state.to_move
    
    @staticmethod
    def actions(state):
        if state.to_move == "B":
            pawns = state.black_pawns
            opponent_pawns = state.white_pawns
            direction = 1
        else:
            pawns = state.white_pawns
            opponent_pawns = state.black_pawns
            direction = -1

        legal_moves = []
        for pawn in pawns:
            row, col = pawn
            if (row + direction, col) not in pawns and 0 <= row + direction <= 2:
                legal_moves.append(("advance", row, col))
            if (row + direction, col - 1) in opponent_pawns and 0 <= row + direction <= 2 and 0 <= col - 1 <= 2:
                legal_moves.append(("capture-left", row, col))
            if (row + direction, col + 1) in opponent_pawns and 0 <= row + direction <= 2 and 0 <= col + 1 <= 2:
                legal_moves.append(("capture-right", row, col))
        return legal_moves
   
    @staticmethod
    def result(state, action):
        if action not in HexapawnState.actions(state):
            return None

        board = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                if state.board[r][c] == "w":
                    board[r][c] = "w"
                elif state.board[r][c] == "b":
                    board[r][c] = "b"

        move_type, row, col = action
        if move_type == "advance":
            if state.to_move == "B":
                board[row + 1][col] = "b"
                board[row][col] = None
                pawns = state.black_pawns.copy()
                pawns.discard((row, col))
                pawns.add((row + 1, col))
                state.black_pawns = pawns
                state.to_move = "W"
            else:
                board[row - 1][col] = "w"
                board[row][col] = None
                pawns = state.white_pawns.copy()
                pawns.discard((row, col))
                pawns.add((row - 1, col))
                state.white_pawns = pawns
                state.to_move = "B"
        elif move_type == "capture-left":
            if state.to_move == "B":
                board[row + 1][col - 1] = "b"
                board[row][col] = None
                pawns = state.black_pawns.copy()
                pawns.discard((row, col))
                pawns.add((row + 1, col - 1))
                opp_pawns = state.white_pawns.copy()
                opp_pawns.discard((row + 1, col - 1))
                state.black_pawns = pawns
                state.white_pawns = opp_pawns
                state.to_move = "W"
            else:
                board[row - 1][col + 1] = "w"
                board[row][col] = None
                pawns = state.white_pawns.copy()
                pawns.discard((row, col))
                pawns.add((row - 1, col + 1))
                opp_pawns = state.black_pawns.copy()
                opp_pawns.discard((row - 1, col + 1))
                state.white_pawns = pawns
                state.black_pawns = opp_pawns
                state.to_move = "B"
        elif move_type == "capture-right":
            if state.to_move == "B":
                board[row + 1][col + 1] = "b"
                board[row][col] = None
                pawns = state.black_pawns.copy()
                pawns.discard((row, col))
                pawns.add((row + 1, col + 1))
                opp_pawns = state.white_pawns.copy()
                opp_pawns.discard((row + 1, col + 1))
                state.black_pawns = pawns
                state.white_pawns = opp_pawns
                state.to_move = "W"
            else:
                board[row - 1][col - 1] = "w"
                board[row][col] = None
                pawns = state.white_pawns.copy()
                pawns.discard((row, col))
                pawns.add((row - 1, col - 1))
                opp_pawns = state.black_pawns.copy()
                opp_pawns.discard((row - 1, col - 1))
                state.white_pawns = pawns
                state.black_pawns = opp_pawns
                state.to_move = "B"
        state.board = board
        return HexapawnState(board, state.to_move)
    
    @staticmethod
    def is_terminal(state):
        # Check if any of the pawns reached the opposite side of the board
        for (row, col) in state.white_pawns:
            if row == 0:
                return True, "W"
        for (row, col) in state.black_pawns:
            if row == 2:
                return True, "B"
        # Check if any player has no pawns left
        if HexapawnState.actions(state):
            return False, None
        # Check the opp pawn able to move or not
        if not HexapawnState.actions(state):
            if state.to_move == "B":
                return True, "W"
            elif state.to_move == "W":
                return True, "B"
        if not state.white_pawns:
            return True, "B"
        if not state.black_pawns:
            return True, "W"
        # Otherwise the game is not terminal
        return False, None

    @staticmethod
    def utility(state):
        game_over, winner = HexapawnState.is_terminal(state)
        if not game_over:
            return 0
        elif winner == "W":
            return 1
        elif winner == "B":
            return -1
        else:
            return 0

init = HexapawnState([['b','b','b'],
        [None, None, None],
        ['w','w','w']])


def minimax_search(game, state):
    value, action = max_value(game, state)
    return action

def max_value(game, state):
    if game.is_terminal(state):
        return game.utility(state), None
    v = float('-inf')
    actions = game.actions(state)
    # best_actions = []
    for a in actions:
        v2, a2 = min_value(game, game.result(state, a))
        if v2 > v:
            v, action = v2, a
            # best_actions = [a]
        # elif v2 == v:
        #     best_actions.append(a)
    return v, action

def min_value(game, state):
    if game.is_terminal(state):
        return game.utility(state), None
    v = float('inf')
    actions = game.actions(state)
    # best_actions = []
    for a in actions:
        v2, a2 = max_value(game, game.result(state, a))
        if v2 < v:
            v, action = v2, a
            # best_actions = [a]
        # elif v2 == v:
        #     best_actions.append(a)
    return v, action
print(HexapawnState.result(init,('advance',0,0)))
# print(minimax_search(init, init))