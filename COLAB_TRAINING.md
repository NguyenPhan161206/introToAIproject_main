# Hướng dẫn huấn luyện ANN trên Google Colab

## Bước 1: Sinh dữ liệu local

```bash
cd gomoku9x9_ai
pip install -r requirements.txt
python3 data/selfplay_data.py --num-games 500 --depth 1
```

Kết quả: `data/X_data.npy` (board states) + `data/y_data.npy` (heuristic scores)

> Với depth=1: ~2 phút. depth=2: ~1.5 giờ (chất lượng hơn).

## Bước 2: Upload lên Google Drive

Tạo folder `gomoku_data` trên Drive, upload:

| File | Vai trò |
|------|---------|
| `data/X_data.npy` | Đầu vào ANN (81 features) |
| `data/y_data.npy` | Nhãn (heuristic score, đã scale [-1,1]) |
| `network/model.py` | Định nghĩa ANN |

## Bước 3: Mở Colab và chạy

1. Vào [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU T4**
3. Tạo cell mới, paste code sau:

```python
# Cell 1: Mount Drive + cài thư viện
from google.colab import drive
drive.mount('/content/drive')

!pip install numpy torch onnx onnxruntime scikit-learn joblib
```

```python
# Cell 2: Load data và model
import numpy as np
import torch
import torch.nn as nn
import joblib
import sys
sys.path.insert(0, '/content/drive/MyDrive/gomoku_data')

from model import HeuristicPredictor

X = np.load('/content/drive/MyDrive/gomoku_data/X_data.npy')
y = np.load('/content/drive/MyDrive/gomoku_data/y_data.npy')
print(f'Data: X {X.shape}, y {y.shape}')
```

```python
# Cell 3: Train/val split + StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).view(-1, 1)

print(f'Train: {len(X_train)}, Val: {len(X_val)}')
```

```python
# Cell 4: Huấn luyện
model = HeuristicPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

best_val = float('inf')
for epoch in range(200):
    model.train()
    perm = torch.randperm(len(X_train_t))
    epoch_loss = 0
    for i in range(0, len(X_train_t), 128):
        idx = perm[i:i+128]
        loss = criterion(model(X_train_t[idx]), y_train_t[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    if epoch % 20 == 0:
        print(f'Epoch {epoch}: train_loss={epoch_loss/(len(X_train_t)//128):.6f}, val_loss={val_loss:.6f}')

print(f'Best val_loss: {best_val:.6f}')
```

```python
# Cell 5: Export ONNX + Scaler
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

dummy = torch.randn(1, 81)
torch.onnx.export(
    model, dummy, 'heuristic_predictor.onnx',
    input_names=['board_input'],
    output_names=['heuristic_score'],
    dynamic_axes={'board_input': {0: 'batch_size'}}
)

joblib.dump(scaler, 'scaler.pkl')
print('Exported: best_model.pth, heuristic_predictor.onnx, scaler.pkl')
```

```python
# Cell 6: Tải về máy
from google.colab import files
files.download('best_model.pth')
files.download('heuristic_predictor.onnx')
files.download('scaler.pkl')
```

## Bước 4: Copy vào local project

Tạo folder `models/` và copy 3 file vừa tải vào:

```
gomoku9x9_ai/models/
├── best_model.pth
├── heuristic_predictor.onnx
└── scaler.pkl
```

## Bước 5: Chạy web game

```bash
python3 web/app.py
```

Khi ANN model được load, dòng sau sẽ hiện:
```
[Web] ANN model loaded successfully
```

Chọn "ANN (Neural Network)" trong dropdown heuristic để dùng.

## Lưu ý

- **ONNX** là file duy nhất cần cho inference, `best_model.pth` chỉ để train tiếp
- `scaler.pkl` phải cùng folder với `.onnx` vì `ann_predictor.py` load nó để chuẩn hóa đầu vào
- Nếu muốn train chất lượng hơn: tăng `--depth 2` và `--num-games 1000` ở bước 1
