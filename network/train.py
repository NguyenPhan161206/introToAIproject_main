import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from network.model import HeuristicPredictor


def train(data_dir='../data', models_dir='../models', epochs=200, batch_size=128, lr=0.001):
    os.makedirs(models_dir, exist_ok=True)

    print('[Train] Loading data...')
    X = np.load(os.path.join(data_dir, 'X_data.npy'))
    y = np.load(os.path.join(data_dir, 'y_data.npy'))
    print(f'  X: {X.shape}, y: {y.shape}')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f'  Train: {len(X_train)}, Val: {len(X_val)}')

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    model = HeuristicPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print('[Train] Starting training...')
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f'  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}')

    print(f'[Train] Best val_loss: {best_val_loss:.6f}')

    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pth')))
    model.eval()

    dummy = torch.randn(1, 81)
    onnx_path = os.path.join(models_dir, 'heuristic_predictor.onnx')
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['board_input'],
        output_names=['heuristic_score'],
        dynamic_axes={'board_input': {0: 'batch_size'}},
        large_model_threshold=0
    )
    print(f'[Train] ONNX model saved to {onnx_path}')

    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f'[Train] Scaler saved to {scaler_path}')

    print('[Train] Done!')
    return model


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    train(data_dir=data_dir, models_dir=models_dir, epochs=200, batch_size=128)
