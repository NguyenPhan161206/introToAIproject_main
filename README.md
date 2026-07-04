# Gomoku 9×9 AI — ANN Heuristic vs Traditional Heuristic

Minh họa **sự giao thoa giữa AI hiện đại và thuật toán truyền thống**: thay thế hàm lượng giá (heuristic function) viết tay bằng một mạng nơ-ron nhân tạo (ANN) đã huấn luyện trong thuật toán **Minimax + Alpha-Beta Pruning** cho trò chơi **Gomoku (Caro) 9×9**.

---

## Giới thiệu

### Mục tiêu đồ án

1. **Triển khai Minimax + Alpha-Beta Pruning** cho game Gomoku 9×9
2. **Xây dựng heuristic truyền thống** dựa trên đếm pattern (pattern-based scoring)
3. **Huấn luyện một ANN** (MLP 3 lớp) để dự đoán điểm heuristic từ trạng thái bàn cờ
4. **So sánh hiệu năng** giữa heuristic truyền thống và ANN khi tích hợp vào cùng thuật toán tìm kiếm
5. **Xây dựng web game** (Flask + HTML/CSS/JS) để người dùng trải nghiệm và so sánh trực quan

### Tại sao Gomoku 9×9?

| Tiêu chí | Connect Four 7×6 | Gomoku 9×9 |
|----------|-----------------|------------|
| Branching factor | ~4 (thấp) | ~40-60 (cao) |
| Minimax depth khả thi | depth 8-10 | depth 2-3 |
| Heuristic cũ | Gần tối ưu | Còn thiếu sót |
| ANN cải thiện | 5-10% | 20-40% |
| Tính trực quan | Bàn nhỏ, ít wow | Bàn 9×9 như cờ thật |

Gomoku 9×9 là điểm cân bằng: bàn đủ lớn để Minimax truyền thống **không thể duyệt sâu**, heuristic cũ **không đủ mạnh**, và ANN có thể thể hiện rõ ưu thế.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                     GOMOKU 9×9 AI                           │
├───────────────────┬─────────────────────────────────────────┤
│   Game Engine     │  game/board.py          (Logic bàn cờ) │
│   (Python)        │  game/heuristic.py      (Heuristic cũ) │
│                   │  game/alpha_beta.py     (Minimax + AB)  │
├───────────────────┼─────────────────────────────────────────┤
│   ANN Training    │  network/model.py       (Định nghĩa ANN)│
│   (PyTorch/Colab) │  network/train.py       (Huấn luyện)   │
│                   │  data/selfplay_data.py  (Sinh dữ liệu)  │
├───────────────────┼─────────────────────────────────────────┤
│   Inference       │  inference/ann_predictor.py (ONNX)     │
│   (ONNX Runtime)  │  inference/compare.py   (So sánh AI)   │
├───────────────────┼─────────────────────────────────────────┤
│   Web Game        │  web/app.py             (Flask server) │
│   (Flask + JS)    │  web/templates/         (Giao diện)    │
│                   │  web/static/js/         (Client logic) │
└───────────────────┴─────────────────────────────────────────┘
```

---

## Cơ sở lý thuyết

### 1. Minimax + Alpha-Beta Pruning

**Minimax** là thuật toán tìm kiếm trên cây trò chơi, giả định đối thủ luôn chọn nước đi tối ưu:
- Nút MAX: người chơi hiện tại chọn nước có điểm cao nhất
- Nút MIN: đối thủ chọn nước có điểm thấp nhất

**Alpha-Beta Pruning** tối ưu Minimax bằng cách cắt tỉa các nhánh không cần thiết:
- `α (alpha)`: giá trị tốt nhất mà MAX có thể đạt được
- `β (beta)`: giá trị tốt nhất mà MIN có thể đạt được
- Nếu `α ≥ β`, dừng duyệt nhánh hiện tại

Với Gomoku 9×9, branching factor ~40-60, depth=2-3 là tối đa.

### 2. Heuristic truyền thống (Pattern-based scoring)

Heuristic truyền thống tính điểm bằng cách duyệt toàn bộ bàn cờ theo 4 hướng, tìm các đoạn quân liên tiếp và gán điểm dựa trên:

| Pattern | Mô tả | Điểm (tấn công) | Điểm (phòng thủ) |
|---------|-------|----------------|------------------|
| `_XXXXX_` | 5 liên tiếp (thắng) | 1.000.000 | — |
| `_XXXX_` | Open four (2 đầu mở) | 100.000 | -90.000 |
| `XXXX_` | Half four (1 đầu mở) | 10.000 | -9.000 |
| `_XXX_` | Open three (2 đầu mở) | 5.000 | -4.500 |
| `XXX__` | Half three (1 đầu mở) | 500 | -450 |

**Hạn chế**: Heuristic cũ chỉ nhìn từng hướng riêng lẻ, không phát hiện được pattern kết hợp nhiều hướng (vd: tạo cùng lúc 2 đường open-three).

### 3. ANN Heuristic (Multi-Layer Perceptron)

**Kiến trúc mạng:**

```
Input: board flatten (81,)    ← giá trị 0 (trống), 1 (X), 2 (O)
       │
   ┌───┴───┐
   │ FC 256 │ → ReLU → BatchNorm → Dropout(0.3)
   ├───────┤
   │ FC 128 │ → ReLU → BatchNorm → Dropout(0.3)
   ├───────┤
   │ FC 64  │ → ReLU → BatchNorm
   ├───────┤
   │ FC 1   │ → Tanh
   └───┬───┘
       │
Output: score ∈ [-1, 1]       ← dự đoán điểm heuristic
```

**Tại sao MLP thay vì CNN?** Với bàn 9×9 (81 ô), MLP đủ để học pattern cơ bản. CNN sẽ mạnh hơn nhưng cần nhiều dữ liệu hơn.

**Huấn luyện:**
- Dữ liệu: 20K-50K `(board_state, heuristic_score)` từ self-play
- Loss: MSE (Mean Squared Error)
- Optimizer: AdamW + CosineAnnealingLR
- Output scaled về [-1, 1] bằng `tanh(score/10000)`

---

## Hướng dẫn cài đặt

### Yêu cầu

- Python 3.8+
- pip

### Bước 1: Clone và cài dependencies

```bash
cd gomoku9x9_ai
pip install -r requirements.txt
```

### Bước 2: Sinh dữ liệu huấn luyện (tùy chọn, có thể dùng dữ liệu có sẵn)

```bash
python data/selfplay_data.py
```

Mặc định sinh 500 ván self-play với depth=2 → ~20K-50K samples.
Thời gian: ~15-30 phút trên CPU.

### Bước 3: Huấn luyện ANN

**Cách 1: Train local (CPU, chậm hơn)**

```bash
python network/train.py
```

**Cách 2: Train trên Google Colab (khuyến nghị)**

1. Upload thư mục `gomoku9x9_ai` lên Google Drive
2. Mở `network/train_colab.ipynb` trên Colab
3. Chọn Runtime → Change runtime type → GPU T4
4. Chạy toàn bộ notebook

Sau khi train, tải các file sau về thư mục `models/`:
- `heuristic_predictor.onnx`
- `scaler.pkl`

### Bước 4: Chạy so sánh AI

```bash
python inference/compare.py
```

So sánh ANN vs Traditional heuristic qua 5 cặp đấu (mỗi cặp 30-50 ván).

### Bước 5: Chạy web game

```bash
python web/app.py
```

Mở trình duyệt tại `http://localhost:5000`

---

## Hướng dẫn sử dụng Web Game

### Giao diện

- **Bàn cờ 9×9**: Click vào ô trống để đặt quân đen (bạn)
- **Bảng điều khiển bên phải**: Cấu hình AI, tỉ số, trạng thái

### Chế độ chơi

1. **Người vs AI**: Click ô bất kỳ, AI (quân trắng) tự động trả lời
2. **AI vs AI (Auto)**: Nhấn nút "AI vs AI" để xem hai AI tự đấu

### Cấu hình AI

- **Heuristic**: Chọn "ANN (Neural Network)" hoặc "Truyền thống (Pattern)"
- **Độ sâu Minimax**: 1-4 (depth càng cao, AI càng mạnh nhưng chậm hơn)

### Thông tin hiển thị

- **Thời gian AI**: Thời gian AI suy nghĩ cho mỗi nước (ms)
- **Số nước**: Tổng số nước đã đánh
- **Tỉ số**: Thắng - Thua - Hòa qua các ván

---

## Cấu trúc code chi tiết

### `game/board.py` — Logic bàn cờ

```python
class Board:
    def __init__(self)              # Khởi tạo bàn 9x9
    def place_piece(self, r, c, p)  # Đặt quân
    def get_valid_moves(self)       # Tất cả ô trống
    def get_valid_moves_nearby(r=2) # Ô trống gần quân hiện có (tối ưu)
    def check_win(self, r, c, p)    # Kiểm tra 5 liên tiếp
    def check_any_win(self, p)      # Kiểm tra toàn bộ
    def is_full(self)               # Bàn đầy?
    def undo(self, r, c)            # Hoàn tác (cho Minimax)
    def copy(self)                  # Deep copy
```

### `game/heuristic.py` — Heuristic truyền thống

```python
evaluate_line(count, open_ends)     # Điểm cho 1 đoạn quân
score_position(board, piece)        # Tổng điểm cho 1 bên
heuristic(board, piece)             # score(piece) - score(opponent)
```

### `game/alpha_beta.py` — Minimax + Alpha-Beta

```python
minimax(board, depth, alpha, beta, maximizing, piece, heuristic_func)
get_best_move(board, depth, piece, heuristic_func)
```

Các tối ưu:
- `get_valid_moves_nearby(radius=2)`: giảm branching factor từ ~60 → ~15
- Kiểm tra thắng/thua ngay trước khi gọi heuristic
- Nước đi trung tâm cho nước đầu tiên

### `data/selfplay_data.py` — Sinh dữ liệu

- Hai AI (cùng heuristic cũ) tự đấu
- Mỗi nước: ghi `(board_state. flatten(), heuristic_score)`
- Scale score về [-1, 1] bằng `tanh(score/10000)`

### `network/model.py` — ANN definition

```python
class HeuristicPredictor(nn.Module):
    # Input: (batch, 81)
    # Hidden: 256 → 128 → 64 (ReLU + BatchNorm + Dropout)
    # Output: (batch, 1) với Tanh
```

### `network/train.py` — Huấn luyện

- Train/val split: 85%/15%
- Loss: MSELoss
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Early stopping: save best model theo val_loss
- Export ONNX + scaler

### `inference/ann_predictor.py` — ANN inference

```python
class ANNPredictor:
    def predict(self, board)  # board: np.array (9,9) → score
```

Hàm `ann_heuristic_wrapper(ann)` trả về hàm heuristic tương thích với `get_best_move`.

### `inference/compare.py` — So sánh AI

So sánh 5 cặp:
1. ANN(d=3) vs Traditional(d=3)
2. Traditional(d=3) vs ANN(d=3)  — đổi lượt đi
3. ANN(d=2) vs Traditional(d=3)
4. ANN(d=3) vs Traditional(d=2)
5. ANN(d=3) vs ANN(d=2)

### `web/app.py` — Flask server

API endpoints:
- `GET /api/state` — Trạng thái hiện tại
- `POST /api/player_move` — Người chơi đánh
- `POST /api/ai_move` — AI đánh
- `POST /api/reset` — Reset game
- `POST /api/ai_vs_ai` — AI tự đấu

---

## Kết quả so sánh dự kiến

| Cặp so sánh | Dự đoán | Giải thích |
|------------|---------|------------|
| ANN(d=3) vs Trad(d=3) | ANN thắng 60-70% | ANN học pattern tốt hơn |
| Trad(d=3) vs ANN(d=3) | ANN thắng 55-65% | Lợi thế đi trước, ANN vẫn nhỉnh hơn |
| ANN(d=2) vs Trad(d=3) | Trad thắng 55-60% | Depth quan trọng hơn heuristic |
| ANN(d=3) vs Trad(d=2) | ANN thắng 75-85% | Cả depth và heuristic đều hơn |
| ANN(d=3) vs ANN(d=2) | ANN(d=3) thắng 70-80% | Depth scaling rõ rệt |

---

## Huấn luyện trên Google Colab

### Notebook colab

Tạo file `network/train_colab.ipynb` với nội dung:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install torch numpy onnx onnxruntime scikit-learn joblib

# 3. Copy data
!cp /content/drive/MyDrive/gomoku9x9_ai/data/X_data.npy .
!cp /content/drive/MyDrive/gomoku9x9_ai/data/y_data.npy .

# 4. Copy model definition
!cp /content/drive/MyDrive/gomoku9x9_ai/network/model.py .

# 5. Run training script
!python -c "
import sys
sys.path.insert(0, '.')
from model import HeuristicPredictor
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

X = np.load('X_data.npy')
y = np.load('y_data.npy')

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1,1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).view(-1,1)

model = HeuristicPredictor()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

best_val = float('inf')
for epoch in range(200):
    model.train()
    perm = torch.randperm(len(X_train_t))
    for i in range(0, len(X_train_t), 128):
        idx = perm[i:i+128]
        loss = criterion(model(X_train_t[idx]), y_train_t[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    if epoch % 20 == 0:
        print(f'Epoch {epoch}: val_loss={val_loss.item():.6f}')

# Export ONNX
dummy = torch.randn(1, 81)
torch.onnx.export(model, dummy, 'heuristic_predictor.onnx',
                  input_names=['board_input'],
                  output_names=['heuristic_score'],
                  dynamic_axes={'board_input': {0: 'batch_size'}})

joblib.dump(scaler, 'scaler.pkl')
print('Done!')
"

# 6. Download results
from google.colab import files
files.download('best_model.pth')
files.download('heuristic_predictor.onnx')
files.download('scaler.pkl')
```

---

## Tính năng mở rộng (Optional)

1. **CNN thay vì MLP**: Dùng Conv2D để nhận diện pattern không gian tốt hơn
2. **Iterative self-play**: Dùng ANN vừa train để sinh thêm data, train lại
3. **MCTS thay vì Minimax**: Monte Carlo Tree Search cho chất lượng cao hơn
4. **Đa luồng**: Tăng tốc Minimax bằng parallel search
5. **Opening book**: Lưu các nước đi mạnh ở đầu ván

---

## Tham khảo

- [AlphaZero: Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- [Minimax algorithm - Wikipedia](https://en.wikipedia.org/wiki/Minimax)
- [Alpha-Beta pruning - Wikipedia](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

---

## Tác giả

Đồ án môn học **Nhập môn Trí tuệ nhân tạo** — Ứng dụng ANN thay thế Heuristic truyền thống trong thuật toán Minimax + Alpha-Beta Pruning cho game Gomoku 9×9.
