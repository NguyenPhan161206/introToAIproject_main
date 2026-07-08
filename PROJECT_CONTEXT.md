# Gomoku 9×9 AI — Project Context

## Overview

A Gomoku (Caro) 9×9 game with two AI modes:
- **Traditional heuristic** — pattern-based evaluation (hand-crafted rules)
- **ANN heuristic** — neural network trained to approximate the traditional heuristic

Both AIs use **Minimax + Alpha-Beta pruning** for move search. The ANN is a student model that learns to mimic the traditional heuristic (knowledge distillation).

## Repository Structure

```
gomoku9x9_ai/
├── data/                # Training data (X_data.npy, y_data.npy) + generation script
├── game/                # Core game logic
│   ├── board.py         # Board state (9×9), move validation, win detection
│   ├── heuristic.py     # Traditional pattern-based heuristic (hand-crafted rules)
│   └── alpha_beta.py    # Minimax + Alpha-Beta search with pruning
├── network/             # ANN model definition + training code + Colab notebook
│   ├── model.py         # PyTorch MLP model (256→128→64→Tanh)
│   ├── train.py         # Local training script (export to ONNX)
│   ├── model.ipynb      # Colab notebook for GPU training
│   └── reportP3/        # Training analysis report (R²=0.0286 before fix)
├── inference/           # ANN inference via ONNX Runtime
│   ├── ann_predictor.py # ANNPredictor class + wrapper function
│   └── compare.py       # ANN vs Traditional heuristic comparison
├── models/              # Trained model files (pre-trained)
│   ├── heuristic_predictor.onnx
│   └── scaler.pkl
├── web/                 # Flask web app with browser UI
│   ├── app.py           # Flask server
│   ├── static/          # JS/CSS files
│   └── templates/       # HTML templates
├── venv/                # Python virtual environment
├── COLAB_TRAINING.md    # Guide: train ANN on Google Colab
├── README.md            # Project README
└── requirements.txt     # Python dependencies
```

## Data Pipeline

### Generation (`data/selfplay_data.py`)

Self-play games using Minimax + Alpha-Beta with traditional heuristic:

1. **Random phase**: `random_moves` nước ngẫu nhiên đầu game (đảm bảo đa dạng)
2. **Minimax phase**: Các nước còn lại dùng Minimax depth=1 (fast) hoặc depth=2 (quality)

Mỗi nước đi (sau random phase) ghi lại:
- **X**: Board state flattened (81 phần tử, giá trị {0=empty, 1=Player X, 2=Player O})
- **y**: Heuristic score, scaled qua `tanh(score / 10000)` → range [-1, 1]

Output: `data/X_data.npy`, `data/y_data.npy`

### Training (`network/train.py` / `network/model.ipynb`)

1. StandardScaler chuẩn hóa X
2. Train/val split (85/15)
3. MLP 81→256→128→64→1 với Tanh đầu ra
4. Loss: MSE, Optimizer: AdamW, CosineAnnealingLR
5. Export: ONNX model + Scaler pickle

### Inference (`inference/ann_predictor.py`)

- Load ONNX model + StandardScaler
- Transform board state → predict heuristic score
- Inverse scaling: `arctanh(pred) * 10000`

## Model Architecture

```
Input (81) → Linear(81, 256) → ReLU → BatchNorm → Dropout(0.3)
          → Linear(256, 128) → ReLU → BatchNorm → Dropout(0.3)
          → Linear(128, 64)  → ReLU → BatchNorm
          → Linear(64, 1)    → Tanh → Output [-1, 1]
```

## Game Flow

```
Board state (9×9)
    → get_valid_moves_nearby(radius=2) → candidate moves
    → Minimax(depth, alpha, beta)
        → leaf: heuristic(board, piece)
    → best move → place piece → check win
```

## Traditional Heuristic (`game/heuristic.py`)

Evaluates patterns in 4 directions (horizontal, vertical, 2 diagonals):

| Pattern | Open 2 ends | Open 1 end | Blocked |
|---------|-------------|------------|---------|
| 5 in row | 1,000,000 | 1,000,000 | 1,000,000 |
| 4 in row | 100,000 | 10,000 | 0 |
| 3 in row | 5,000 | 500 | 0 |
| 2 in row | 200 | 50 | 0 |
| 1 in row | 10 | 1 | 0 |

Opponent patterns weighted ×1.1 (defensive bias).

## History & Evolution

- **Commit 819d88d**: Initial project — Gomoku 9×9 with Minimax + Alpha-Beta + Flask
- **Commit e00161c**: Added `random_moves` to data generation for diversity
- **Commit 4f2a539**: Cleaned up duplicate notebooks, fixed ONNX export
- **Commit 98b86da**: Added training analysis with loss curves, R²=0.0286 report
- **Current fix**: Changed from 2→10 random moves, skip recording random-phase states, eliminated 99.8% data duplication

## Current Dataset Status (after fix)

- **19,107 samples**, 0% duplicate (before: 40,500 samples, 99.8% duplicate)
- **9,959 unique y values** (before: 74)
- Ready for Colab training with `network/model.ipynb`

## Key Dependencies

- numpy, torch, onnx, onnxruntime
- scikit-learn (StandardScaler)
- flask (web app)
- joblib (scaler serialization)
