import numpy as np
import onnxruntime as ort
import joblib # thư viện này dùng để lưu trữ và tải các đối tượng Python, chẳng hạn như bộ scaler đã được huấn luyện. Trong trường hợp này, nó được sử dụng để tải bộ scaler từ tệp pickle (scaler.pkl) để chuẩn hóa dữ liệu đầu vào trước khi đưa vào mô hình ONNX.
import os


class ANNPredictor:
    def __init__(self, onnx_path=None, scaler_path=None): #onnx là đường dẫn đến mô hình ONNX đã được huấn luyện, scaler_path là đường dẫn đến bộ scaler đã được lưu trữ. Nếu không cung cấp, chúng sẽ được đặt mặc định trong thư mục models.
        if onnx_path is None:
            onnx_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'heuristic_predictor.onnx'
            )
        if scaler_path is None:
            scaler_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'scaler.pkl'
            )

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f'ONNX model not found at {onnx_path}. '
                'Run network/train.py first or download the pre-trained model.'
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f'Scaler not found at {scaler_path}. '
                'Run network/train.py first.'
            )

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.scaler = joblib.load(scaler_path)

    def predict(self, board):
        flat = board.flatten().astype(np.float32).reshape(1, -1)
        scaled = self.scaler.transform(flat)
        result = self.session.run(None, {self.input_name: scaled})
        return float(result[0][0][0])


def ann_heuristic_wrapper(ann_predictor):
    def ann_heuristic(board_state, piece):
        from game.board import PLAYER_X
        b = board_state.get_board()
        if piece == PLAYER_X:
            raw_score = ann_predictor.predict(b)
        else:
            flipped = np.where(b == 1, 2, np.where(b == 2, 1, 0))
            raw_score = ann_predictor.predict(flipped)
        clipped = np.clip(raw_score, -0.9999, 0.9999)
        score = float(np.arctanh(clipped) * 10000)
        return score
    return ann_heuristic
