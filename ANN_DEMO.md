# Minh chứng sử dụng ANN trong đồ án Gomoku 9×9 AI

## Mục lục

1. [Kiến trúc mạng ANN — `network/model.py`](#1-kiến-trúc-mạng-ann--networkmodelpy)
2. [Huấn luyện ANN — `network/train.py`](#2-huấn-luyện-ann--networktrainpy)
3. [Sinh dữ liệu tự động (Self-play) — `data/selfplay_data.py`](#3-sinh-dữ-liệu-tự-động-self-play--dataselfplay_datapy)
4. [Inference ANN qua ONNX — `inference/ann_predictor.py`](#4-inference-ann-qua-onnx--inferenceann_predictorpy)
5. [Tích hợp ANN vào Minimax + Web App — `game/alpha_beta.py` + `web/app.py`](#5-tích-hợp-ann-vào-minimax--web-app--gamealpha_betapy--webapppy)

---

## 1. Kiến trúc mạng ANN — `network/model.py`

```python
import torch
import torch.nn as nn

INPUT_SIZE = 81
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64


class HeuristicPredictor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),   # (1) 81 → 256
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(0.3),

            nn.Linear(hidden1, hidden2),      # (2) 256 → 128
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(0.3),

            nn.Linear(hidden2, hidden3),      # (3) 128 → 64
            nn.ReLU(),
            nn.BatchNorm1d(hidden3),

            nn.Linear(hidden3, 1),            # (4) 64 → 1
            nn.Tanh()                         # đầu ra trong [-1, 1]
        )

    def forward(self, x):
        return self.net(x)
```

### Giải thích

| Thành phần | Ý nghĩa |
|------------|---------|
| `nn.Linear(81, 256)` | Lớp fully-connected: nhận 81 đầu vào (bảng 9×9 flattened) → 256 neuron ẩn |
| `nn.ReLU()` | Hàm kích hoạt ReLU: `f(x) = max(0, x)`, giúp mạng học phi tuyến |
| `nn.BatchNorm1d` | Chuẩn hóa batch: giúp ổn định và tăng tốc hội tụ khi huấn luyện |
| `nn.Dropout(0.3)` | Dropout 30%: kỹ thuật chống overfitting, ngẫu nhiên tắt 30% neuron |
| `nn.Linear(64, 1)` | Lớp đầu ra: gom 64 đặc trưng thành 1 giá trị điểm số |
| `nn.Tanh()` | Ép đầu ra về [-1, 1], khớp với nhãn đã được scale bằng `tanh` |

**Kiến trúc tổng thể**: MLP 4 lớp `81 → 256 → 128 → 64 → 1`.

**Số tham số**: ~81×256 + 256×128 + 128×64 + 64×1 = **62.272** tham số — đủ nhỏ để inference nhanh.

---

## 2. Huấn luyện ANN — `network/train.py`

```python
from network.model import HeuristicPredictor

def train(data_dir='../data', models_dir='../models', epochs=200, batch_size=128, lr=0.001):
    # Tải dữ liệu
    X = np.load(os.path.join(data_dir, 'X_data.npy'))   # board states (N × 81)
    y = np.load(os.path.join(data_dir, 'y_data.npy'))   # heuristic scores (N,)

    # Chuẩn hóa đầu vào bằng StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Chia train/val: 85% / 15%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Chuyển sang tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    model = HeuristicPredictor()
    criterion = nn.MSELoss()                                    # hàm mất mát: sai số bình phương trung bình
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))                   # xáo trộn dữ liệu

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(X_batch)                               # forward: dự đoán
            loss = criterion(pred, y_batch)                     # tính MSE loss
            loss.backward()                                     # backward: lan truyền ngược
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                                    # cập nhật trọng số

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step()

    # Export sang ONNX
    dummy = torch.randn(1, 81)
    torch.onnx.export(model, dummy, 'heuristic_predictor.onnx',
                      input_names=['board_input'],
                      output_names=['heuristic_score'])
```

### Giải thích

| Bước | Ý nghĩa |
|------|---------|
| `StandardScaler` | Chuẩn hóa đầu vào về trung bình 0, phương sai 1 — giúp ANN hội tụ nhanh hơn |
| `train_test_split(0.15)` | Dành 15% dữ liệu làm tập validation để đánh giá |
| `MSELoss` | Hàm mất mát phù hợp cho bài toán hồi quy (regression): `MSE = mean((y_true - y_pred)²)` |
| `AdamW` | Optimizer cải tiến từ Adam, có weight decay giúp chống overfitting |
| `CosineAnnealingLR` | Giảm learning rate theo chu kỳ hình cos — giúp hội tụ tốt hơn |
| `loss.backward()` | Lan truyền ngược (backpropagation): tính gradient của loss theo từng tham số |
| `optimizer.step()` | Cập nhật trọng số theo gradient đã tính |
| `clip_grad_norm_` | Giới hạn norm gradient ≤ 1.0 — tránh exploding gradient |
| `torch.onnx.export` | Xuất mô hình sang ONNX — cho phép inference không cần PyTorch |

---

## 3. Sinh dữ liệu tự động (Self-play) — `data/selfplay_data.py`

```python
def generate_training_data(num_games=500, depth=1, random_moves=10, output_dir='.'):
    X_list, y_list = [], []

    for game_idx in range(num_games):
        board = Board()
        turn = 0

        while True:
            piece = PLAYER_X if turn % 2 == 0 else PLAYER_O

            if turn < random_moves:
                # Giai đoạn 1: đánh ngẫu nhiên để đa dạng dữ liệu
                move = random.choice(board.get_valid_moves())
            else:
                # Giai đoạn 2: dùng Minimax + heuristic truyền thống
                move = get_best_move(board, depth=depth, piece=piece, heuristic_func=heuristic)

            board.place_piece(r, c, piece)

            if turn >= random_moves:
                # Ghi nhận: board state → heuristic score
                score = heuristic(board, piece)                            # target
                state = board.get_board().flatten().astype(np.float32)     # input
                X_list.append(state)
                y_list.append(score)

            if board.check_win(r, c, piece) or board.is_full():
                break
            turn += 1

    # Scale target về [-1, 1] bằng tanh
    y_scaled = np.tanh(y / 10000.0)

    np.save('X_data.npy', X)      # input: board states (N × 81)
    np.save('y_data.npy', y_scaled)  # target: heuristic đã scale
```

### Giải thích

| Thành phần | Ý nghĩa |
|------------|---------|
| `random_moves=10` | 10 nước đầu mỗi ván đánh ngẫu nhiên — tạo đa dạng trạng thái bàn cờ |
| `heuristic(board, piece)` | Hàm heuristic truyền thống (dựa trên pattern) — đóng vai trò "giáo viên" (teacher) |
| `board.flatten()` | Chuyển bảng 9×9 thành vector 81 phần tử — đầu vào của ANN |
| `np.tanh(y / 10000.0)` | Scale target về [-1, 1] vì Tanh ở đầu ra ANN cũng cho giá trị trong khoảng này |

**Cơ chế Knowledge Distillation**: ANN không tự học chơi cờ, mà học để bắt chước (mimic) heuristic truyền thống. Heuristic truyền thống là "teacher", ANN là "student".

---

## 4. Inference ANN qua ONNX — `inference/ann_predictor.py`

```python
import onnxruntime as ort
import joblib

class ANNPredictor:
    def __init__(self, onnx_path, scaler_path):
        self.session = ort.InferenceSession(onnx_path)       # tải mô hình ONNX
        self.input_name = self.session.get_inputs()[0].name  # lấy tên input
        self.scaler = joblib.load(scaler_path)               # tải scaler

    def predict(self, board):
        flat = board.flatten().astype(np.float32).reshape(1, -1)
        scaled = self.scaler.transform(flat)                 # chuẩn hóa giống lúc train
        result = self.session.run(None, {self.input_name: scaled})
        return float(result[0][0][0])                        # kết quả trong [-1, 1]


def ann_heuristic_wrapper(ann_predictor):
    def ann_heuristic(board_state, piece):
        b = board_state.get_board()
        # Lật board nếu là quân O (ANN chỉ train với góc nhìn quân X)
        flipped = np.where(b == 1, 2, np.where(b == 2, 1, 0))
        raw_score = ann_predictor.predict(flipped if piece != PLAYER_X else b)

        # Giải scale: atanh(clipped) * 10000
        clipped = np.clip(raw_score, -0.9999, 0.9999)
        score = float(np.arctanh(clipped) * 10000)
        return score
    return ann_heuristic
```

### Giải thích

| Thành phần | Ý nghĩa |
|------------|---------|
| `ort.InferenceSession` | Tải mô hình ONNX để chạy inference — không cần PyTorch, nhẹ và nhanh |
| `scaler.transform(flat)` | Chuẩn hóa đầu vào giống hệt lúc huấn luyện (dùng cùng scaler) |
| `self.session.run(...)` | Chạy forward pass trên ONNX Runtime — trả về giá trị trong [-1, 1] |
| `np.arctanh(clipped) * 10000` | **Giải scale**: ANN học output là `tanh(score/10000)`, nên muốn lấy score gốc phải tính `atanh(output) × 10000` |
| `flipped` | Lật quân 1↔2: ANN luôn nhìn board từ góc quân X, nếu lượt quân O thì phải đổi chiều |
| `ann_heuristic_wrapper` | Trả về hàm có **cùng interface** `(board_state, piece) → score` với heuristic truyền thống |

---

## 5. Tích hợp ANN vào Minimax + Web App

### 5a. `game/alpha_beta.py` — Hàm Minimax dùng heuristic_func

```python
def minimax(board, depth, alpha, beta, maximizing, piece, heuristic_func):
    if depth == 0:
        return heuristic_func(board, piece)    # <-- gọi heuristic tại nút lá

    valid_moves = board.get_valid_moves_nearby(radius=2)
    for r, c in valid_moves:
        board.place_piece(r, c, piece)
        score = minimax(board, depth - 1, alpha, beta, not maximizing, piece, heuristic_func)
        board.undo(r, c)
        # cập nhật alpha/beta...
    return best_score


def get_best_move(board, depth, piece, heuristic_func=None):
    if heuristic_func is None:
        heuristic_func = heuristic_trad         # mặc định dùng heuristic truyền thống

    for r, c in valid_moves:
        board.place_piece(r, c, piece)
        score = minimax(board, depth - 1, ..., piece, heuristic_func)  # <-- truyền heuristic_func
        board.undo(r, c)
    return best_move
```

### 5b. `web/app.py` — Chọn heuristic theo chế độ chơi

```python
from inference.ann_predictor import ANNPredictor, ann_heuristic_wrapper

# Tải ANN khi khởi động
ann_predictor = ANNPredictor()
ann_heuristic_fn = ann_heuristic_wrapper(ann_predictor)

def get_heuristic(heuristic_type):
    if heuristic_type == 'ann' and ann_heuristic_fn is not None:
        return ann_heuristic_fn     # <-- ANN heuristic
    return heuristic_trad            # <-- traditional heuristic


# API ai_move: chọn heuristic theo mode
@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    heuristic_type = 'ann' if mode == 'pvai_ann' else 'traditional'
    h_func = get_heuristic(heuristic_type)

    move = get_best_move(board_copy, depth=depth, piece=PLAYER_O, heuristic_func=h_func)
    # ↑ cùng hàm get_best_move, chỉ khác heuristic_func


# API ai_vs_ai: so sánh ANN vs Traditional
@app.route('/api/ai_vs_ai', methods=['POST'])
def ai_vs_ai():
    h1 = get_heuristic(h1_type)    # có thể là ANN
    h2 = get_heuristic(h2_type)    # có thể là Traditional

    # Hai AI đấu với nhau, mỗi bên dùng heuristic khác nhau
    move = get_best_move(b, depth=depth, piece=piece, heuristic_func=h1)
```

### Giải thích

| Thành phần | Ý nghĩa |
|------------|---------|
| `heuristic_func(board, piece)` | Interface chung: cả ANN và Traditional đều có cùng chữ ký hàm |
| Drop-in replacement | Đổi heuristic chỉ bằng cách truyền hàm khác — **không cần sửa Minimax** |
| `get_heuristic('ann')` | Chọn ANN hay Traditional dựa trên mode game |
| `pvai_ann` | Chế độ "Người vs AI (dùng ANN)" — minh chứng ANN thay thế heuristic truyền thống |
| `ai_vs_ai` | Chế độ ANN đấu với Traditional — đánh giá chất lượng ANN |

---

## Tổng kết

Luồng dữ liệu và mô hình ANN trong đồ án:

```
Self-play ──→ Board states ──→ Traditional Heuristic ──→ (input, target)
   ↑                                                          ↓
   |                                             ANN (MLP) huấn luyện
   |                                                          ↓
   |                                            ONNX Export + Scaler
   |                                                          ↓
   └────────────── ANN Predictor ────→ Minimax + Alpha-Beta ──→ Nước đi
```

**ANN là một MLP hồi quy** được huấn luyện bằng supervised learning để bắt chước heuristic truyền thống. Sau đó nó được đóng gói thành ONNX, tích hợp làm hàm đánh giá (heuristic) trong thuật toán Minimax + Alpha-Beta pruning — thay thế hoàn toàn heuristic gốc mà không cần sửa đổi gì ở bộ tìm kiếm.
