import numpy as np
import os
import pickle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'processed_data')


def verify_processed_data():
    print("=" * 50)
    print("VERİ DOĞRULAMA KONTROLÜ")
    print("=" * 50)

    # Eğitim verisini yükleme
    train_file = os.path.join(DATA_PATH, 'train.npz')
    if not os.path.exists(train_file):
        raise FileNotFoundError("train.npz dosyası bulunamadı.")

    train_data = np.load(train_file)
    x_train, y_train = train_data['x'], train_data['y']

    print(f"[*] X_train boyutu: {x_train.shape} (Beklenen: N, 12, 207, 2)")
    print(f"[*] Y_train boyutu: {y_train.shape} (Beklenen: N, 12, 207, 2)")

    # NaN Kontrolü
    has_nan = np.isnan(x_train).any()
    print(f"[*] Veride NaN (Kayıp) değer var mı?: {has_nan}")

    # Normalizasyon Sınır Kontrolü (Sadece Kanal 0 - Hız)
    x_min = x_train[..., 0].min()
    x_max = x_train[..., 0].max()
    print(f"[*] X_train Hız (Kanal 0) Min Değeri: {x_min:.4f}")
    print(f"[*] X_train Hız (Kanal 0) Max Değeri: {x_max:.4f}")

    # Zaman Bilgisi Sınır Kontrolü (Sadece Kanal 1 - Zaman)
    t_min = x_train[..., 1].min()
    t_max = x_train[..., 1].max()
    print(f"[*] X_train Zaman (Kanal 1) Min Değeri: {t_min:.4f}")
    print(f"[*] X_train Zaman (Kanal 1) Max Değeri: {t_max:.4f}")

    # Scaler Parametreleri
    with open(os.path.join(DATA_PATH, 'scaler_params.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(f"[*] Kaydedilen Scaler Parametreleri: Mean={scaler['mean']:.4f}, Std={scaler['std']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    verify_processed_data()