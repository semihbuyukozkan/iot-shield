import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'plots')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 6)})


def load_data():

    h5_path = os.path.join(DATA_DIR, 'metr-la.h5')
    pkl_path = os.path.join(DATA_DIR, 'adj_mx.pkl')

    # 1. DOSYA KONTROLÜ
    if not os.path.exists(h5_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"\n[HATA] Veri dosyaları bulunamadı!\n"
            f"Lütfen 'metr-la.h5' ve 'adj_mx.pkl' dosyalarını manuel indirip şu klasöre atın:\n"
            f"{DATA_DIR}"
        )

    print("\n" + "=" * 50)
    print("1. VERİ YÜKLEME VE KONTROL")
    print("=" * 50)

    # 2. VERİ YÜKLEME
    try:
        df = pd.read_hdf(h5_path)
        print(f"Trafik verisi (.h5) okundu.")
    except Exception as e:
        print(f".h5 dosyası bozuk veya okunamıyor. Lütfen dosyayı yenileyin.")
        raise e

    try:
        with open(pkl_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
        print(f"Komşuluk matrisi (.pkl) okundu.")
    except Exception as e:
        print(f".pkl dosyası bozuk. Lütfen dosyayı yenileyin.")
        raise e

    # 3. UYUMLULUK KONTROLÜ (SANITY CHECK)
    data_nodes = df.shape[1]  # Veri setindeki sütun sayısı (Sensörler)
    graph_nodes = adj_mx.shape[0]  # Matristeki satır sayısı (Düğümler)

    print("-" * 30)
    print(f"Veri Seti Sensör Sayısı : {data_nodes}")
    print(f"Harita Sensör Sayısı    : {graph_nodes}")

    if data_nodes == graph_nodes:
        print("Veri ve Harita birbiriyle tam uyumlu.")
    else:
        raise ValueError(
            f"\nUYUMSUZLUK TESPİT EDİLDİ"
            f"Veri setinde {data_nodes} sensör var ama haritada {graph_nodes} sensör var.\n"
            "Dosyalar farklı veri setlerine ait olabilir."
        )
    print("-" * 30)

    return df, pkl_path


def print_thesis_statistics(df):

    print("\n" + "=" * 50)
    print("2. TEZ İÇİN İSTATİSTİKLER (METR-LA)")
    print("=" * 50)

    # Genel Boyutlar
    n_samples, n_nodes = df.shape

    # İstatistiksel Hesaplamalar
    mean_val = df.values.mean()
    std_val = df.values.std()
    min_val = df.values.min()
    max_val = df.values.max()

    # Sıfır Değer Analizi (Sensör hatası veya tam durma)
    zeros = (df.values == 0).sum()
    total_cells = n_samples * n_nodes
    zero_ratio = (zeros / total_cells) * 100

    # Veri Seti Ayrımı (Zaman Serisi için Kronolojik Ayrım)
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.1)
    test_size = n_samples - train_size - val_size

    print(f"[*] Toplam Zaman Adımı (Samples): {n_samples:,}")
    print(f"[*] Sensör Sayısı (Nodes)       : {n_nodes}")
    print(f"[*] Veri Başlangıç Tarihi       : {df.index.min()}")
    print(f"[*] Veri Bitiş Tarihi           : {df.index.max()}")
    print("-" * 30)
    print(f"[*] Ortalama Hız (Mean)         : {mean_val:.2f} mph")
    print(f"[*] Standart Sapma (Std)        : {std_val:.2f} mph")
    print(f"[*] Minimum Değer               : {min_val:.2f} mph")
    print(f"[*] Maksimum Değer              : {max_val:.2f} mph")
    print(f"[*] Sıfır Değer Sayısı (Oran)   : {zeros:,} ({zero_ratio:.2f}%)")
    print("-" * 30)
    print("VERİ SETİ AYRIMI (Kronolojik - 70/10/20):")
    print(f"[*] Eğitim Seti (Training)      : {train_size:,} örnek")
    print(f"[*] Doğrulama Seti (Validation) : {val_size:,} örnek")
    print(f"[*] Test Seti (Test)            : {test_size:,} örnek")
    print("=" * 50 + "\n")


def plot_temporal_patterns(df):

    print("Zamansal Örüntüler Çiziliyor...")

    # Karakteristik davranış gösteren örnek sensörler (Rastgele 10-13 arası)
    sample_sensors = df.columns[10:13]

    # --- DÜZELTME BAŞLANGIÇ ---
    # Pandas'ın bazı sürümlerinde string slicing hatası (UnboundLocalError) oluşabildiği için
    # Boolean Masking yöntemi kullanıyoruz. Bu yöntem daha kararlıdır.
    start_date = "2012-03-01"
    end_date = "2012-03-07"

    # İndekslerin bu tarihler arasında olduğu satırları seçiyoruz.
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df.loc[mask, sample_sensors]
    # --- DÜZELTME BİTİŞ ---

    plt.figure(figsize=(15, 6))
    for sensor in sample_sensors:
        plt.plot(subset.index, subset[sensor], label=f"Sensör {sensor}", alpha=0.8, linewidth=1.5)

    plt.title("Trafik Hızının Zamansal Değişimi (1 Hafta)", fontsize=14)
    plt.ylabel("Hız (mph)", fontsize=12)
    plt.xlabel("Zaman", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, '1_temporal_patterns.png')
    plt.savefig(save_path, dpi=300)
    print(f"-> Kaydedildi: {save_path}")
    plt.show()

def plot_spatial_correlation(df):

    print("Mekansal Korelasyon Haritası Çiziliyor...")

    corr_matrix = df.corr()

    # En yüksek korelasyona sahip 15 sensörü seçtik.
    target_sensor = df.columns[0]
    top_correlated = corr_matrix[target_sensor].sort_values(ascending=False).head(15).index

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_correlated].corr(), annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title("Sensörler Arası Korelasyon Matrisi (Seçilmiş Bölge)", fontsize=14)

    save_path = os.path.join(OUTPUT_DIR, '2_spatial_correlation.png')
    plt.savefig(save_path, dpi=300)
    print(f"-> Kaydedildi: {save_path}")


def plot_network_topology(pkl_path):
    print("Ağ Topolojisi Çiziliyor...")

    with open(pkl_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')

    # 1. Matris Yapısı (Spy Plot)
    plt.figure(figsize=(8, 8))
    plt.spy(adj_mx, markersize=1, precision=0.1, color='darkblue')
    plt.title("Komşuluk Matrisi (Adjacency Matrix) Seyrekliği", fontsize=14)
    plt.xlabel("Sensör ID")
    plt.ylabel("Sensör ID")
    plt.savefig(os.path.join(OUTPUT_DIR, '3_adj_matrix_structure.png'), dpi=300)

    # 2. Fiziksel Graf (NetworkX)
    try:
        # İlk 20 sensörlük alt küme (Görsel netliği için)
        subset_size = 20
        adj_subset = adj_mx[:subset_size, :subset_size]
        G = nx.from_numpy_array(adj_subset)

        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray',
                node_size=600, font_size=10, font_weight='bold')
        plt.title(f"Sensör Topolojisi Örneği (İlk {subset_size} Düğüm)", fontsize=14)

        save_path_graph = os.path.join(OUTPUT_DIR, '3_network_topology.png')
        plt.savefig(save_path_graph, dpi=300)
        print(f"-> Kaydedildi: {save_path_graph}")

    except Exception as e:
        print(f"NetworkX grafiği çizilirken hata: {e}")


def plot_speed_distribution(df):

    print("Hız Dağılımı Çiziliyor...")

    plt.figure(figsize=(10, 6))
    sns.histplot(df.values.flatten(), bins=50, kde=True, color='purple', stat="density")
    plt.title("Tüm Ağdaki Hız Değerlerinin Dağılımı", fontsize=14)
    plt.xlabel("Hız (mph)", fontsize=12)
    plt.ylabel("Yoğunluk", fontsize=12)

    save_path = os.path.join(OUTPUT_DIR, '4_speed_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"-> Kaydedildi: {save_path}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_traffic, pkl_file_path = load_data()

    print_thesis_statistics(df_traffic)
    plot_temporal_patterns(df_traffic)
    plot_spatial_correlation(df_traffic)
    plot_network_topology(pkl_file_path)
    plot_speed_distribution(df_traffic)

    print("\n" + "=" * 50)
    print("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI")
    print(f"Grafikler şu klasörde: {OUTPUT_DIR}")
    print("=" * 50)

    plt.show()