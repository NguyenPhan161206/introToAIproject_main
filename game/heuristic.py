import numpy as np
from .board import ROWS, COLS, WIN_LEN, EMPTY, PLAYER_X, PLAYER_O

WIN_SCORE = 1_000_000


def score_line(line, piece): #line là một danh sách các ô trên bảng, piece là người chơi hiện tại (PLAYER_X hoặc PLAYER_O). Hàm này sẽ tính điểm cho dòng đó dựa trên số lượng quân cờ liên tiếp của người chơi hiện tại và đối thủ.
    opponent = PLAYER_O if piece == PLAYER_X else PLAYER_X # dòng này là để xác định đối thủ của người chơi hiện tại. Nếu người chơi hiện tại là PLAYER_X, thì đối thủ sẽ là PLAYER_O và ngược lại. Điều này giúp trong việc đánh giá các dòng trên bảng để tính điểm cho cả người chơi và đối thủ.
    score = 0
    i = 0
    while i < len(line):
        if line[i] == EMPTY:
            i += 1
            continue
        p = line[i]
        count = 0
        while i < len(line) and line[i] == p:
            count += 1
            i += 1
        open_ends = 0
        if i - count - 1 >= 0 and line[i - count - 1] == EMPTY:
            open_ends += 1
        if i < len(line) and line[i] == EMPTY:
            open_ends += 1

        if count >= WIN_LEN:
            eval_score = WIN_SCORE
        elif count == 4:
            if open_ends == 2:
                eval_score = 100_000
            elif open_ends == 1:
                eval_score = 10_000
            else:
                eval_score = 0
        elif count == 3:
            if open_ends == 2:
                eval_score = 5_000
            elif open_ends == 1:
                eval_score = 500
            else:
                eval_score = 0
        elif count == 2:
            if open_ends == 2:
                eval_score = 200
            elif open_ends == 1:
                eval_score = 50
            else:
                eval_score = 0
        elif count == 1:
            if open_ends == 2:
                eval_score = 10
            elif open_ends == 1:
                eval_score = 1
            else:
                eval_score = 0
        else:
            eval_score = 0

        if p == piece:
            score += eval_score
        elif p == opponent:
            score -= eval_score * 1.1
    return score


def score_position(board, piece): # Hàm này tính điểm tổng thể của bảng dựa trên các dòng ngang, dọc và chéo. Nó sử dụng hàm score_line để đánh giá từng dòng và cộng điểm lại để đưa ra một giá trị tổng thể cho vị trí hiện tại của người chơi.
    score = 0
    for r in range(ROWS):
        for c in range(COLS - WIN_LEN + 1):
            line = [board[r, c + i] for i in range(WIN_LEN)]
            score += score_line(line, piece)
    for c in range(COLS):
        for r in range(ROWS - WIN_LEN + 1):
            line = [board[r + i, c] for i in range(WIN_LEN)]
            score += score_line(line, piece)
    for r in range(ROWS - WIN_LEN + 1):
        for c in range(COLS - WIN_LEN + 1):
            line = [board[r + i, c + i] for i in range(WIN_LEN)]
            score += score_line(line, piece)
    for r in range(WIN_LEN - 1, ROWS):
        for c in range(COLS - WIN_LEN + 1):
            line = [board[r - i, c + i] for i in range(WIN_LEN)]
            score += score_line(line, piece)
    return score


def heuristic(board_obj, piece): # Hàm này là hàm heuristic chính, nó nhận vào một đối tượng bảng (board_obj) và người chơi hiện tại (piece). Nó sẽ tính điểm tổng thể của bảng cho người chơi hiện tại và trừ đi điểm tổng thể của đối thủ. Kết quả trả về là một giá trị số, đại diện cho mức độ ưu thế của người chơi hiện tại so với đối thủ.
    if isinstance(board_obj, np.ndarray):
        board = board_obj
    else:
        board = board_obj.get_board()
    return score_position(board, piece) - score_position(board, 3 - piece)
