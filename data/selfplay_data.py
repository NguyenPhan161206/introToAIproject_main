import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.board import Board, PLAYER_X, PLAYER_O
from game.alpha_beta import get_best_move
from game.heuristic import heuristic


def generate_training_data(num_games=500, depth=1, random_moves=10, output_dir='.'):
    """
    Generate training data via self-play with Minimax + Alpha-Beta.

    Args:
        num_games: Number of self-play games
        depth: Minimax search depth (1=fast but noisy, 2=slower but higher quality)
        random_moves: Number of initial random moves per game (default 10)
        output_dir: Directory to save .npy files

    Note:
        depth=1: ~0.2s/game, good for quick iteration (19K samples / 2000 games)
        depth=2: ~11s/game, better data quality (>35K samples / 500 games)
    """
    X_list = []
    y_list = []

    for game_idx in range(num_games):
        board = Board()
        turn = 0

        while True:
            piece = PLAYER_X if turn % 2 == 0 else PLAYER_O

            if turn < random_moves:
                valid_moves = board.get_valid_moves()
                move = random.choice(valid_moves) if valid_moves else None
            else:
                move = get_best_move(board, depth=depth, piece=piece, heuristic_func=heuristic)
            if move is None:
                break

            r, c = move
            board.place_piece(r, c, piece)

            if turn >= random_moves:
                score = heuristic(board, piece)
                state = board.get_board().flatten().astype(np.float32)
                X_list.append(state)
                y_list.append(score)

            if board.check_win(r, c, piece):
                break
            if board.is_full():
                break

            turn += 1

        if (game_idx + 1) % 50 == 0:
            print(f'[Self-play] Game {game_idx + 1}/{num_games} completed')
            if (game_idx + 1) % 200 == 0:
                elapsed_so_far = 0  # approximate
                print(f'  Estimated time remaining...')

    if not X_list:
        raise ValueError(
            'No training data generated. '
            'The AI could not produce any games. '
            'Check get_best_move logic or increase board size.'
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    y_scaled = np.tanh(y / 10000.0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_data.npy'), X)
    np.save(os.path.join(output_dir, 'y_data.npy'), y_scaled)
    print(f'[Self-play] Saved {len(X)} samples to {output_dir}/')
    print(f'  X shape: {X.shape}, y shape: {y_scaled.shape}')
    print(f'  y min: {y_scaled.min():.4f}, y max: {y_scaled.max():.4f}')

    return X, y_scaled


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate self-play training data')
    parser.add_argument('--num-games', type=int, default=500, help='Number of self-play games')
    parser.add_argument('--depth', type=int, default=1,
                        help='Minimax depth (1=fast, 2=better but slower)')
    parser.add_argument('--random-moves', type=int, default=10,
                        help='Number of initial random moves per game (0=no random)')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    print(f'Generating {args.num_games} games at depth {args.depth} with {args.random_moves} random moves...')
    generate_training_data(num_games=args.num_games, depth=args.depth,
                           random_moves=args.random_moves, output_dir=output_dir)
