import matplotlib.pyplot as plt
from skimage import data, transform
import numpy as np
from algorithms import SegmentationBenchmarker  # Az önce oluşturduğumuz dosyayı çağırıyoruz


def main():
    print("=== Karmaşık Ağlar Projesi: Görüntü Bölütleme Analizi ===")

    # 1. VERİ SEÇİMİ
    # Kahve fincanı resmi, doku ve kontrast açısından iyi bir testtir.
    original_img = data.coffee()

    # İşlem hızını artırmak için biraz küçültelim (Opsiyonel)
    img = transform.resize(original_img, (400, 600), anti_aliasing=True)
    # 0-255 arasına çekelim (Scikit-image bazen 0-1 döndürür)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Benchmarker sınıfını başlat
    benchmarker = SegmentationBenchmarker(img)

    # 2. DENEYLERİ ÇALIŞTIR

    # A) Klasik Yöntem (K-Means)
    # 3 gruba ayır (Örn: Fincan, Tabak, Arka Plan)
    benchmarker.run_kmeans(n_clusters=3)

    # B) Bizim Yöntemimiz (Graph-Based)
    # SLIC süper piksel sayısı grafiğin düğüm sayısını belirler
    benchmarker.run_spectral_slic(n_segments=400, compactness=30)

    # 3. SONUÇLARI GÖRSELLEŞTİR VE KIYASLA
    plot_results(img, benchmarker.results, benchmarker.timings)


def plot_results(original, results, timings):
    """
    Sonuçları yan yana profesyonel bir grafikte gösterir.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Orijinal Resim
    axes[0].imshow(original)
    axes[0].set_title("Orijinal Görüntü\n(Input)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # K-Means Sonucu
    t_kmeans = timings['K-Means']
    axes[1].imshow(results['K-Means'])
    axes[1].set_title(f"Klasik Yöntem: K-Means\nSüre: {t_kmeans:.2f} sn", fontsize=12)
    axes[1].set_xlabel("Sadece renk kümelemesi yapar.\nUzamsal ilişkiyi bilmez.", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Graph-Based Sonucu
    t_graph = timings['Graph-Based']
    axes[2].imshow(results['Graph-Based'])
    axes[2].set_title(f"Proje Yöntemi: Graph-Based (N-Cuts)\nSüre: {t_graph:.2f} sn", fontsize=12, color='darkblue')
    axes[2].set_xlabel("Ağ teorisi kullanır.\nNesne bütünlüğünü korur.", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()