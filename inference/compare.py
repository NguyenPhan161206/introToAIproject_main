import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.board import Board, PLAYER_X, PLAYER_O
from game.alpha_beta import get_best_move
from game.heuristic import heuristic as heuristic_trad

try:
    from inference.ann_predictor import ANNPredictor, ann_heuristic_wrapper
    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False
    ANNPredictor = None
    ann_heuristic_wrapper = None


def play_match(heuristic_func_x, heuristic_func_o, num_games=50,
               depth_x=3, depth_o=3, verbose=False):
    results = {'X': 0, 'O': 0, 'draw': 0}
    total_moves = 0

    for game_idx in range(num_games):
        board = Board()
        turn = 0
        move_count = 0

        while True:
            current_piece = PLAYER_X if turn % 2 == 0 else PLAYER_O
            h_func = heuristic_func_x if current_piece == PLAYER_X else heuristic_func_o
            depth = depth_x if current_piece == PLAYER_X else depth_o

            move = get_best_move(board, depth=depth, piece=current_piece, heuristic_func=h_func)
            if move is None:
                results['draw'] += 1
                break

            r, c = move
            board.place_piece(r, c, current_piece)
            move_count += 1

            if board.check_win(r, c, current_piece):
                winner = 'X' if current_piece == PLAYER_X else 'O'
                results[winner] += 1
                break
            if board.is_full():
                results['draw'] += 1
                break

            turn += 1

        total_moves += move_count

        if verbose and (game_idx + 1) % 10 == 0:
            print(f'  Match {game_idx + 1}/{num_games}: X={results["X"]}, O={results["O"]}, draw={results["draw"]}')

    avg_moves = total_moves / num_games
    return results, avg_moves


def run_comparison(num_games=50, depth=3):
    print('=' * 60)
    print('GOMOKU 9x9 AI COMPARISON')
    print(f'Games per match: {num_games}, Default depth: {depth}')
    print('=' * 60)

    global ANN_AVAILABLE
    ann = None
    ann_heuristic = heuristic_trad
    
    if ANN_AVAILABLE:
        try:
            ann = ANNPredictor()
            ann_heuristic = ann_heuristic_wrapper(ann)
            print('[OK] ANN model loaded')
        except Exception as e:
            ANN_AVAILABLE = False
            print(f'[WARN] ANN model not loadable: {e}')
            print('[WARN] Only traditional heuristic comparisons will run')

    d = depth
    d1 = max(1, d - 1)
    if ANN_AVAILABLE:
        comparisons = [
            (f'ANN (d={d}) vs Traditional (d={d})', ann_heuristic, heuristic_trad, d, d),
            (f'Traditional (d={d}) vs ANN (d={d})', heuristic_trad, ann_heuristic, d, d),
            (f'ANN (d={d1}) vs Traditional (d={d})', ann_heuristic, heuristic_trad, d1, d),
            (f'ANN (d={d}) vs Traditional (d={d1})', ann_heuristic, heuristic_trad, d, d1),
            (f'ANN (d={d}) vs ANN (d={d1})', ann_heuristic, ann_heuristic, d, d1),
        ]
    else:
        comparisons = [
            (f'Traditional (d={d}) vs Traditional (d={d})', heuristic_trad, heuristic_trad, d, d),
            (f'Traditional (d={d1}) vs Traditional (d={d})', heuristic_trad, heuristic_trad, d1, d),
        ]

    for name, h_x, h_o, d_x, d_o in comparisons:
        print(f'\n--- {name} ---')
        start = time.time()

        results, avg_moves = play_match(
            h_x, h_o,
            num_games=num_games,
            depth_x=d_x, depth_o=d_o,
            verbose=True
        )
        elapsed = time.time() - start

        total = results['X'] + results['O'] + results['draw']
        print(f'\n  Results:')
        print(f'    X (first player, d={d_x}): {results["X"]} wins ({results["X"]/total*100:.1f}%)')
        print(f'    O (second player, d={d_o}): {results["O"]} wins ({results["O"]/total*100:.1f}%)')
        print(f'    Draws: {results["draw"]} ({results["draw"]/total*100:.1f}%)')
        print(f'    Avg moves/game: {avg_moves:.1f}')
        print(f'    Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s/game)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare AI heuristics')
    parser.add_argument('--games', type=int, default=30, help='Number of games per match')
    parser.add_argument('--depth', type=int, default=3, help='Search depth')
    args = parser.parse_args()
    run_comparison(num_games=args.games, depth=args.depth)
