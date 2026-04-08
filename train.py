import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pickle
import time

from graphwavenetmodel import GraphWaveNet

# Dizin Yolları
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def load_data(batch_size):
    print("[*] Veriler yükleniyor ve tensör boyutları ayarlanıyor...")
    train_npz = np.load(os.path.join(PROCESSED_DIR, 'train.npz'))
    val_npz = np.load(os.path.join(PROCESSED_DIR, 'val.npz'))

    # X verisini Graph WaveNet formatına (N, C, V, T) çevirme
    # (N, 12, 207, 2) -> (N, 2, 207, 12)
    x_train = torch.tensor(train_npz['x']).permute(0, 3, 2, 1).float()
    x_val = torch.tensor(val_npz['x']).permute(0, 3, 2, 1).float()

    # Y verisinden sadece Hız (0. kanal) tahmin edileceği için filtrelenir
    # (N, 12, 207, 2) -> (N, 12, 207, 1)
    y_train = torch.tensor(train_npz['y'][..., 0:1]).float()
    y_val = torch.tensor(val_npz['y'][..., 0:1]).float()

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # Bellek yönetimi için veriler batch'lere bölünür
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_adj():
    pkl_path = os.path.join(DATA_DIR, 'adj_mx.pkl')
    with open(pkl_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    return [torch.tensor(adj_mx).float()]


def masked_mae_loss(preds, labels, null_val=0.0):
    # Eksik verileri (0.0 mph) kayıp (loss) hesaplamasından çıkarır
    mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def train():
    print("=" * 50)
    print("GRAPH WAVENET EĞİTİM SÜRECİ")
    print("=" * 50)

    # Donanım ayarı
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Kullanılan Donanım: {device}")

    # Hiperparametreler
    epochs = 50
    batch_size = 64
    learning_rate = 0.001

    train_loader, val_loader = load_data(batch_size)
    adj_matrices = [m.to(device) for m in load_adj()]

    print("[*] Model derleniyor...")
    model = GraphWaveNet(
        device=device,
        num_nodes=207,
        in_dim=2,
        out_dim=12,
        supports=adj_matrices
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    print("[*] Eğitim başlıyor...\n")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)

            loss = masked_mae_loss(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation (Doğrulama) Aşaması
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = masked_mae_loss(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        end_time = time.time()

        print(
            f"Epoch: {epoch + 1:02d}/{epochs} | Süre: {end_time - start_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Modelin en iyi versiyonunu diske kaydetme
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'graphwavenet_best.pth'))
            print("   -> Gelişim tespit edildi. Model kaydedildi.")

    print("\n" + "=" * 50)
    print("EĞİTİM BAŞARIYLA TAMAMLANDI")
    print(f"En iyi model 'saved_models' klasörüne kaydedildi.")
    print("=" * 50)


if __name__ == '__main__':
    train()