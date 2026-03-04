import os
import pandas as pd
import numpy as np
import pickle

'''
METR-LA Veri Seti Ön İşleme (Preprocessing) Adımları
Bu kodla, veriyi Graph WaveNet mimarisinin beklediği giriş formatına dönüştürüyoruz.
'''

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'processed_data')


SEQ_LEN_X = 12  # Geçmiş 12 adım (60 dakikayı bölmek için)
SEQ_LEN_Y = 12  # Gelecek 12 adım
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1


def load_and_add_features(h5_path):
    print("1. Veri yükleniyor ve öznitelik mühendisliği (Time of Day) uygulanıyor...")
    df = pd.read_hdf(h5_path)

    # Hız verisi (N, 207) -> (N, 207, 1) formatına genişletildi.
    speed_data = np.expand_dims(df.values, axis=-1)

    # Zaman indeksi kullanılarak [0, 1) aralığında "Günün Saati" hesapladık.
    df_index = df.index
    tod = (df_index.hour * 60 + df_index.minute) / 1440.0

    # Zaman verisi, tüm sensörler için aynı olacak şekilde genişletilir: (N, 207, 1)
    tod_data = np.tile(tod.values.reshape(-1, 1, 1), (1, speed_data.shape[1], 1))

    # Hız ve Zaman verileri birleştirilir: Nihai boyut (N, 207, 2)
    combined_data = np.concatenate([speed_data, tod_data], axis=-1)

    print(f"   -> Oluşturulan ham veri matrisi boyutu: {combined_data.shape}")
    return combined_data


def split_data(data):
    print("2. Veri seti eğitim, doğrulama ve test olarak bölünüyor (70/10/20)...")
    num_samples = data.shape[0]
    train_steps = int(num_samples * TRAIN_RATIO)
    val_steps = int(num_samples * VAL_RATIO)

    train_data = data[:train_steps]
    val_data = data[train_steps: train_steps + val_steps]
    test_data = data[train_steps + val_steps:]

    print(f"   -> Eğitim Seti Boyutu: {train_data.shape}")
    print(f"   -> Doğrulama Seti Boyutu: {val_data.shape}")
    print(f"   -> Test Seti Boyutu: {test_data.shape}")

    return train_data, val_data, test_data


def normalize_data(train, val, test):
    print("3. Z-Score Normalizasyonu uygulanıyor (Sadece Hız verisi üzerinde)...")
    # Sızıntıyı önlemek için sadece eğitim setinin istatistikleri kullandık
    # Kanal 0: Hız verisi, Kanal 1: Zaman verisi (Zaten [0,1] aralığında)
    mean = train[..., 0].mean()
    std = train[..., 0].std()

    print(f"   -> Hesaplanan Ortalama (Mean): {mean:.4f}, Standart Sapma (Std): {std:.4f}")

    # Kopyalama uyarısını önlemek için array'ler kopyaladık
    train_norm = np.copy(train)
    val_norm = np.copy(val)
    test_norm = np.copy(test)

    train_norm[..., 0] = (train_norm[..., 0] - mean) / std
    val_norm[..., 0] = (val_norm[..., 0] - mean) / std
    test_norm[..., 0] = (test_norm[..., 0] - mean) / std

    scaler_params = {'mean': mean, 'std': std}
    return train_norm, val_norm, test_norm, scaler_params


def generate_sliding_windows(data, x_len, y_len):
    x_list, y_list = [], []
    total_len = data.shape[0]

    for i in range(total_len - x_len - y_len + 1):
        x = data[i: i + x_len, ...]
        y = data[i + x_len: i + x_len + y_len, ...]
        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


def process_and_save():
    h5_path = os.path.join(DATA_DIR, 'metr-la.h5')

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"[HATA] {h5_path} bulunamadı.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("VERİ ÖN İŞLEME SÜRECİ BAŞLIYOR")
    print("=" * 50)

    # 1. Yükleme ve Feature Ekleme
    data = load_and_add_features(h5_path)

    # 2. Bölme
    train_data, val_data, test_data = split_data(data)

    # 3. Normalizasyon
    train_norm, val_norm, test_norm, scaler_params = normalize_data(train_data, val_data, test_data)

    # 4. Kayan Pencere (Sliding Window)
    print("4. Kayan pencere metodu ile tensör boyutları oluşturuluyor (X:12, Y:12)...")
    x_train, y_train = generate_sliding_windows(train_norm, SEQ_LEN_X, SEQ_LEN_Y)
    x_val, y_val = generate_sliding_windows(val_norm, SEQ_LEN_X, SEQ_LEN_Y)
    x_test, y_test = generate_sliding_windows(test_norm, SEQ_LEN_X, SEQ_LEN_Y)

    print(f"   -> X_train boyutu: {x_train.shape}, Y_train boyutu: {y_train.shape}")
    print(f"   -> X_val boyutu  : {x_val.shape}, Y_val boyutu  : {y_val.shape}")
    print(f"   -> X_test boyutu : {x_test.shape}, Y_test boyutu : {y_test.shape}")


    print("5. İşlenmiş veriler '.npz' ve '.pkl' formatında kaydediliyor...")
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'train.npz'), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'test.npz'), x=x_test, y=y_test)

    with open(os.path.join(OUTPUT_DIR, 'scaler_params.pkl'), 'wb') as f:
        pickle.dump(scaler_params, f)

    print("\n" + "=" * 50)
    print("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI")
    print(f"Çıktı klasörü: {OUTPUT_DIR}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    process_and_save()