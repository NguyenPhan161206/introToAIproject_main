import numpy as np
from .board import Board, ROWS, COLS, EMPTY, PLAYER_X, PLAYER_O
from .heuristic import heuristic as heuristic_trad

MIN_SCORE = -float('inf')
MAX_SCORE = float('inf')

def minimax(board, depth, alpha, beta, maximizing, piece, heuristic_func):
    if depth == 0:
        return heuristic_func(board, piece)

    opponent = 3 - piece

    if board.check_any_win(piece):
        return 1_000_000 * (depth + 1)
    if board.check_any_win(opponent):
        return -1_000_000 * (depth + 1)
    if board.is_full():
        return 0

    valid_moves = board.get_valid_moves_nearby(radius=2)
    if not valid_moves:
        valid_moves = board.get_valid_moves()

    if maximizing:
        value = MIN_SCORE
        for r, c in valid_moves:
            board.place_piece(r, c, piece)
            score = minimax(board, depth - 1, alpha, beta, False, piece, heuristic_func)
            board.undo(r, c)
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value
    else:
        value = MAX_SCORE
        for r, c in valid_moves:
            board.place_piece(r, c, opponent)
            score = minimax(board, depth - 1, alpha, beta, True, piece, heuristic_func)
            board.undo(r, c)
            if score < value:
                value = score
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value


def get_best_move(board, depth, piece, heuristic_func=None):
    if heuristic_func is None:
        heuristic_func = heuristic_trad

    valid_moves = board.get_valid_moves_nearby(radius=2)
    if not valid_moves:
        valid_moves = board.get_valid_moves()
    if not valid_moves:
        return None

    opponent = 3 - piece

    for r, c in valid_moves:
        board.place_piece(r, c, piece)
        if board.check_win(r, c, piece):
            board.undo(r, c)
            return (r, c)
        board.undo(r, c)

    for r, c in valid_moves:
        board.place_piece(r, c, opponent)
        if board.check_win(r, c, opponent):
            board.undo(r, c)
            return (r, c)
        board.undo(r, c)

    first_move = (ROWS // 2, COLS // 2)
    if first_move in valid_moves and np.count_nonzero(board.get_board()) == 0:
        return first_move

    best_move = None
    best_score = MIN_SCORE

    for r, c in valid_moves:
        board.place_piece(r, c, piece)
        score = minimax(board, depth - 1, MIN_SCORE, MAX_SCORE, False, piece, heuristic_func)
        board.undo(r, c)
        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move
