import torch
from graphwavenetmodel import GraphWaveNet


def run_sanity_check():
    print("=" * 50)
    print("MODEL DONANIM VE BOYUT KONTROLÜ (SANITY CHECK)")
    print("=" * 50)

    # 1. Evrensel Donanım (Device) Kontrolü
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[*] Donanım: NVIDIA GPU (CUDA) başarıyla algılandı.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[*] Donanım: Apple Silicon GPU (MPS) başarıyla algılandı.")
    else:
        device = torch.device("cpu")
        print("[*] Donanım: Özel GPU bulunamadı, standart CPU kullanılacak.")

    # 2. Hiperparametreler ve Modelin Başlatılması
    num_nodes = 207  # METR-LA sensör sayısı
    in_dim = 2  # Kanal Sayısı: Hız ve Zaman (Time of Day)
    out_dim = 12  # Tahmin edilecek gelecek adım sayısı
    batch_size = 32  # Test için rastgele yığın boyutu
    seq_len = 12  # Girdi olarak verilecek geçmiş adım sayısı

    print("\n[*] Graph WaveNet modeli oluşturuluyor...")
    model = GraphWaveNet(
        device=device,
        num_nodes=num_nodes,
        in_dim=in_dim,
        out_dim=out_dim
    ).to(device)

    # 3. Dummy (Sahte) Veri Üretimi
    # PyTorch beklenen tensör formatı: (Batch_Size, Channels, Nodes, Sequence_Length)
    dummy_input = torch.randn(batch_size, in_dim, num_nodes, seq_len).to(device)
    print(f"[*] Girdi Tensör Boyutu : {dummy_input.shape}")

    # 4. İleri Yayılım (Forward Pass) Testi
    print("\n[*] İleri yayılım (forward pass) testi başlatılıyor...")
    try:
        model.eval()  # Dropout gibi eğitim özelliklerini devre dışı bırakmak için
        with torch.no_grad():
            output = model(dummy_input)

        print("[*] İleri yayılım başarılı!")
        print(f"[*] Çıktı Tensör Boyutu : {output.shape}")

        # Beklenen çıktı boyutu: (Batch_Size, Out_Dim, Nodes, 1)
        expected_shape = (batch_size, out_dim, num_nodes, 1)
        if output.shape == expected_shape:
            print(f"[*] Boyut Kontrolü      : BAŞARILI (Beklenen boyut ile eşleşti)")
        else:
            print(f"[!] Boyut Kontrolü      : BAŞARISIZ (Beklenen: {expected_shape})")

    except Exception as e:
        print(f"\n[HATA] İleri yayılım sırasında bir hata oluştu:\n{e}")

    print("=" * 50)


if __name__ == "__main__":
    run_sanity_check()