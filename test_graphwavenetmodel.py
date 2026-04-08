import torch
import numpy as np
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader

from graphwavenetmodel import GraphWaveNet

# Dizin Yolları
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')


# Maskeli metrik fonksiyonları (Kayıp veriyi -0.0 mph- göz ardı eder)
def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    return torch.mean(loss * mask)


def masked_rmse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(preds - labels)
    return torch.sqrt(torch.mean(loss * mask))


def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / (labels + 1e-5)  # Sıfıra bölme engeli
    return torch.mean(loss * mask)


def load_test_data(batch_size=64):
    test_npz = np.load(os.path.join(PROCESSED_DIR, 'test.npz'))
    x_test = torch.tensor(test_npz['x']).permute(0, 3, 2, 1).float()
    y_test = torch.tensor(test_npz['y'][..., 0:1]).float()

    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def load_adj():
    with open(os.path.join(DATA_DIR, 'adj_mx.pkl'), 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    return [torch.tensor(adj_mx).float()]


def test():
    print("=" * 50)
    print("MODEL TEST VE DEĞERLENDİRME SÜRECİ")
    print("=" * 50)

    # Donanım
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_loader = load_test_data()
    adj_matrices = [m.to(device) for m in load_adj()]

    # Scaler parametrelerini yükle (Z-Score ters dönüşümü için)
    with open(os.path.join(PROCESSED_DIR, 'scaler_params.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean, std = scaler['mean'], scaler['std']

    # Model Kurulumu ve Ağırlıkların Yüklenmesi
    print("[*] En iyi model ağırlıkları yükleniyor...")
    model = GraphWaveNet(device=device, num_nodes=207, in_dim=2, out_dim=12, supports=adj_matrices).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, 'graphwavenet_best.pth'), map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("[*] Test verisi üzerinde tahminler yapılıyor...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # Tensörleri birleştir ve gerçek mph değerlerine dönüştür
    all_preds = torch.cat(all_preds, dim=0) * std + mean
    all_labels = torch.cat(all_labels, dim=0) * std + mean

    print("\n[*] Metrikler Hesaplanıyor...")
    mae = masked_mae(all_preds, all_labels).item()
    rmse = masked_rmse(all_preds, all_labels).item()
    mape = masked_mape(all_preds, all_labels).item() * 100  # Yüzdelik formata çevrilir

    print(f"[*] MAE  (Ortalama Mutlak Hata)    : {mae:.2f} mph")
    print(f"[*] RMSE (Kök Ortalama Kare Hata)  : {rmse:.2f} mph")
    print(f"[*] MAPE (Ortalama Mutlak Yüzde Hata): %{mape:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    test()