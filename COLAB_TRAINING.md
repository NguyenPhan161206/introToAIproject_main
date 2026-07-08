# Hướng dẫn huấn luyện ANN trên Google Colab

## Bước 1: Sinh dữ liệu local (đã cải thiện)

```bash
cd gomoku9x9_ai
source venv/bin/activate
pip install -r requirements.txt
python3 data/selfplay_data.py --num-games 2000 --depth 1 --random-moves 10
```

Kết quả: `data/X_data.npy` + `data/y_data.npy`

> **Cải tiến so với trước:**
> - `--random-moves 10`: 10 nước random đầu giúp các ván đấu đa dạng (trước chỉ 2 → 99.8% duplicate)
> - `--num-games 2000`: 19,000+ mẫu unique (trước 500 game chỉ được 81 board state)
> - Chỉ ghi nhận board state sau 10 nước random (bỏ qua nhiễu)

Thời gian: ~7 phút với depth=1, ~1.5 giờ nếu muốn chất lượng cao hơn (depth=2).

## Bước 2: Mở notebook trên Colab

1. Upload file `network/model.ipynb` lên [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU T4**
3. Chạy từng cell theo thứ tự

### Notebook sẽ tự động:
- Cell 1: Mount Google Drive (tuỳ chọn, có thể bỏ qua)
- Cell 2: Cài thư viện
- **Cell 3: Upload `X_data.npy` và `y_data.npy` từ máy bạn** (dùng `files.upload()`)
- Cell 4: Định nghĩa model ANN (MLP 256→128→64 → Tanh)
- Cell 5: Load data, train/val split, StandardScaler, huấn luyện 200 epochs
- Cell 6: Export ONNX + Scaler
- Cell 7: Tải `heuristic_predictor.onnx` + `scaler.pkl` về máy

## Bước 3: Copy model vào local project

```bash
cp heuristic_predictor.onnx models/
cp scaler.pkl models/
```

## Bước 4: Chạy web game

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
