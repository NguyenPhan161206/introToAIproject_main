import numpy as np
from copy import deepcopy

ROWS = 9
COLS = 9
WIN_LEN = 5
EMPTY = 0
PLAYER_X = 1  # black, goes first
PLAYER_O = 2  # white

class Board:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)

    def place_piece(self, row, col, player):
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return False
        if self.board[row, col] != EMPTY:
            return False
        self.board[row, col] = player
        return True

    def get_valid_moves(self):
        return list(zip(*np.where(self.board == EMPTY)))

    def get_valid_moves_nearby(self, radius=2):
        occupied = np.argwhere(self.board != EMPTY)
        if len(occupied) == 0:
            return [(ROWS // 2, COLS // 2)]
        moves_set = set()
        for r, c in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and self.board[nr, nc] == EMPTY:
                        moves_set.add((nr, nc))
        return list(moves_set)

    def check_win(self, row, col, player):
        if self.board[row, col] != player:
            return False
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= WIN_LEN:
                return True
        return False

    def check_any_win(self, player):
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r, c] == player and self.check_win(r, c, player):
                    return True
        return False

    def is_full(self):
        return np.all(self.board != EMPTY)

    def undo(self, row, col):
        self.board[row, col] = EMPTY

    def copy(self):
        return deepcopy(self)

    def get_board(self):
        return self.board.copy()

    def __str__(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        lines.append('  ' + ' '.join(str(c) for c in range(COLS)))
        for r in range(ROWS):
            line = str(r) + ' ' + ' '.join(symbols[self.board[r, c]] for c in range(COLS))
            lines.append(line)
        return '\n'.join(lines)
