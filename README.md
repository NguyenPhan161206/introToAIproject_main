# Gomoku 9×9 AI — ANN Heuristic vs Traditional Heuristic

Gomoku (Caro) 9×9 với AI dùng Minimax + Alpha-Beta, cho phép chọn giữa heuristic truyền thống (pattern-based) và ANN (neural network) đã huấn luyện.

## Yêu cầu

- Python 3.10+
- pip

## Cài đặt & Chạy

```bash
# 1. Tạo virtual environment (Ubuntu 24+ bắt buộc)
python3 -m venv venv
source venv/bin/activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy web app
python3 web/app.py
```

Mở trình duyệt tại **http://localhost:5000**

## Chế độ chơi

| Mode | Mô tả |
|------|-------|
| **PvAI (ANN)** | Người vs AI dùng ANN heuristic (cần model trong `models/`) |
| **PvAI (Trad)** | Người vs AI dùng heuristic truyền thống |
| **PvP** | 2 người chơi trên cùng máy |
| **AI vs AI** | ANN vs Traditional tự động |

## Cấu hình

- **Độ sâu Minimax**: 1-4 (mặc định 3). Càng sâu → càng mạnh nhưng chậm hơn.

## Huấn luyện ANN

### 1. Sinh dữ liệu self-play

```bash
python3 data/selfplay_data.py --num-games 2000 --depth 1 --random-moves 10
```
→ Tạo `data/X_data.npy` (~6MB) + `data/y_data.npy` (~0.1MB), ~19K mẫu unique

> **Gợi ý:** `--random-moves 10` giúp các ván đấu đa dạng (tránh duplicate). Tăng `--depth 2` nếu muốn chất lượng cao hơn.

### 2. Train trên Google Colab

1. Upload `network/model.ipynb` lên [colab.research.google.com](https://colab.research.google.com)
2. Chạy từng cell, upload `X_data.npy` + `y_data.npy` khi được yêu cầu
3. Tải `heuristic_predictor.onnx` + `scaler.pkl` về

### 3. Copy model vào project

```bash
cp heuristic_predictor.onnx models/
cp scaler.pkl models/
```

Khởi động lại server → nút **PvAI (ANN)** sẽ khả dụng.

## Cấu trúc thư mục

```
gomoku9x9_ai/
├── data/            # Dữ liệu training + script sinh dữ liệu
├── game/            # Board, heuristic, alpha-beta
├── inference/       # ANN predictor (ONNX Runtime)
├── network/         # Định nghĩa model + train script + Colab notebook
├── web/             # Flask app + frontend
└── models/          # ONNX model + scaler (cần tự train)
```

## Lưu ý

- Server chạy ở `http://0.0.0.0:5000`
- Nếu không có model ANN, app tự động fallback sang traditional heuristic
- Luôn chạy `source venv/bin/activate` trước khi dùng Python
