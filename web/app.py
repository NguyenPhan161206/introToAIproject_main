import sys
import os # os là dung de thao tác với hệ thống tệp và đường dẫn, ví dụ như xác định vị trí của các tệp mô hình và scaler.
# scaler là một công cụ chuẩn hóa dữ liệu, giúp đưa các giá trị đầu vào về cùng một thang đo, điều này rất quan trọng khi sử dụng mô hình học máy. Trong trường hợp này, scaler được sử dụng để chuẩn hóa dữ liệu đầu vào trước khi đưa vào mô hình ONNX.
import time
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, render_template, request, jsonify
from game.board import Board, ROWS, COLS, PLAYER_X, PLAYER_O
from game.alpha_beta import get_best_move
from game.heuristic import heuristic as heuristic_trad

app = Flask(__name__)

board = Board()
game_over = False
winner = None
mode = 'pvai_trad'
board_lock = threading.Lock()
ann_predictor = None
ann_heuristic_fn = None
ann_error = None


def load_ann():
    global ann_predictor, ann_heuristic_fn, ann_error
    try:
        from inference.ann_predictor import ANNPredictor, ann_heuristic_wrapper
        ann_predictor = ANNPredictor()
        ann_heuristic_fn = ann_heuristic_wrapper(ann_predictor)
        print('[Web] ANN model loaded successfully')
        return True
    except Exception as e:
        ann_error = str(e)
        print(f'[Web] Could not load ANN model: {e}')
        print('[Web] Falling back to traditional heuristic only')
        return False

ANN_AVAILABLE = load_ann()

def get_heuristic(heuristic_type):
    if heuristic_type == 'ann' and ann_heuristic_fn is not None:
        return ann_heuristic_fn
    return heuristic_trad


def validate_move_data(data):
    """Validate row, col, depth from JSON request."""
    if data is None:
        return None, None, None, 'Request body is empty'
    try:
        row = int(data.get('row', -1))
        col = int(data.get('col', -1))
    except (ValueError, TypeError):
        return None, None, None, 'Row and col must be numbers'
    if not (0 <= row < ROWS and 0 <= col < COLS):
        return None, None, None, f'Position ({row}, {col}) out of bounds'
    return row, col, None, None


def get_current_turn():
    """Determine whose turn it is based on piece count."""
    piece_count = np.count_nonzero(board.get_board())
    return PLAYER_X if piece_count % 2 == 0 else PLAYER_O


def handle_game_end(detected_winner):
    """Set game_over and winner, return consistent JSON response."""
    global game_over, winner
    game_over = True
    winner = detected_winner


def board_response(last_move=None, thinking_time_ms=None):
    """Build standardized board response."""
    resp = {
        'board': board.get_board().tolist(),
        'game_over': game_over,
        'winner': winner,
    }
    if last_move:
        resp['last_move'] = [int(last_move[0]), int(last_move[1])]
    if thinking_time_ms is not None:
        resp['thinking_time_ms'] = round(thinking_time_ms, 1)
    if not game_over:
        current = get_current_turn()
        if mode == 'pvp':
            resp['current_turn'] = int(current)
            resp['turn_label'] = 'Đen' if current == PLAYER_X else 'Trắng'
        else:
            resp['current_turn'] = 'player' if current == PLAYER_X else 'ai'
            resp['turn_label'] = 'Bạn' if current == PLAYER_X else 'AI'
    return resp


@app.route('/')
def index():
    return render_template('index.html', ann_available=ANN_AVAILABLE, ann_error=ann_error)


@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(board_response())


@app.route('/api/mode', methods=['GET'])
def get_mode():
    return jsonify({'mode': mode, 'ann_available': ANN_AVAILABLE, 'ann_error': ann_error})


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    global mode, game_over, winner, board
    with board_lock:
        data = request.json
        if not data or 'mode' not in data:
            return jsonify({'error': 'Missing mode field'}), 400
        new_mode = data['mode']
        if new_mode not in ('pvai_ann', 'pvai_trad', 'pvp'):
            return jsonify({'error': 'Mode must be pvai_ann, pvai_trad, or pvp'}), 400
        if new_mode == 'pvai_ann' and not ANN_AVAILABLE:
            return jsonify({'error': 'ANN model not available'}), 400
        mode = new_mode
        board = Board()
        game_over = False
        winner = None
        return jsonify({'status': 'ok', 'mode': mode})


@app.route('/api/player_move', methods=['POST'])
def player_move():
    global game_over, winner
    data = request.json
    row, col, _, error = validate_move_data(data)
    if error:
        return jsonify({'error': error}), 400

    with board_lock:
        if game_over:
            return jsonify({'error': 'Game is over'}), 400

        current_turn = get_current_turn()

        if mode.startswith('pvai') and current_turn != PLAYER_X:
            return jsonify({'error': 'Not your turn — AI is thinking'}), 400

        piece = current_turn

        if not board.place_piece(row, col, piece):
            return jsonify({'error': 'Cell is already occupied'}), 400

        if board.check_win(row, col, piece):
            if mode.startswith('pvai'):
                handle_game_end('player')
            else:
                handle_game_end('black' if piece == PLAYER_X else 'white')
        elif board.is_full():
            handle_game_end('draw')

        return jsonify(board_response(last_move=(row, col)))


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    global game_over, winner
    data = request.json
    try:
        depth = int(data.get('depth', 3))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid depth value'}), 400
    if not (1 <= depth <= 4):
        depth = 3

    heuristic_type = 'ann' if mode == 'pvai_ann' else 'traditional'
    h_func = get_heuristic(heuristic_type)

    with board_lock:
        if game_over:
            return jsonify({'error': 'Game is over'}), 400

        if mode == 'pvp':
            return jsonify({'error': 'AI not available in PvP mode'}), 400

        current_turn = get_current_turn()
        if current_turn != PLAYER_O:
            return jsonify({'error': 'Not AI turn'}), 400

        board_copy = board.copy()
        start = time.time()
        move = get_best_move(board_copy, depth=depth, piece=PLAYER_O, heuristic_func=h_func)
        elapsed = (time.time() - start) * 1000

        if move is None:
            handle_game_end('draw')
            return jsonify(board_response(thinking_time_ms=elapsed))

        row, col = move
        board.place_piece(row, col, PLAYER_O)

        if board.check_win(row, col, PLAYER_O):
            handle_game_end('ai')
        elif board.is_full():
            handle_game_end('draw')

        return jsonify(board_response(last_move=(row, col), thinking_time_ms=elapsed))


@app.route('/api/reset', methods=['POST'])
def reset():
    global board, game_over, winner
    with board_lock:
        board = Board()
        game_over = False
        winner = None
        return jsonify({'status': 'ok'})


@app.route('/api/ai_vs_ai', methods=['POST'])
def ai_vs_ai():
    data = request.json
    h1_type = data.get('heuristic1', 'ann') if ANN_AVAILABLE else 'traditional'
    h2_type = data.get('heuristic2', 'traditional')
    try:
        depth = int(data.get('depth', 3))
    except (ValueError, TypeError):
        depth = 3

    h1 = get_heuristic(h1_type)
    h2 = get_heuristic(h2_type)

    b = Board()
    turn = 0
    moves = []

    while True:
        piece = PLAYER_X if turn % 2 == 0 else PLAYER_O
        h = h1 if piece == PLAYER_X else h2

        move = get_best_move(b, depth=depth, piece=piece, heuristic_func=h)
        if move is None:
            return jsonify({'winner': 'draw', 'moves': moves})

        r, c = move
        b.place_piece(r, c, piece)
        moves.append({
            'row': int(r), 'col': int(c),
            'player': 'black' if piece == PLAYER_X else 'white'
        })

        if b.check_win(r, c, piece):
            return jsonify({
                'winner': 'black' if piece == PLAYER_X else 'white',
                'moves': moves
            })
        if b.is_full():
            return jsonify({'winner': 'draw', 'moves': moves})

        turn += 1


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
