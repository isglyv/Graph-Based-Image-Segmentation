import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from skimage.transform import resize
from skimage import data

# 1. Veri Yükleme ve Ön İşleme (Downsampling )
# Örnek olarak scikit-image içindeki 'coins'resmini kullanıyoruz.
original_img = data.coins()

# İşlem maliyetini düşürmek için resmi küçültüyoruz
# Orijinal boyutun %20'si.
res_factor = 0.20
img_small = resize(original_img, (int(original_img.shape[0] * res_factor),
                                  int(original_img.shape[1] * res_factor)),
                   anti_aliasing=True)

# Görüntüyü 0-1 arasına normalize et
img_small = img_small / np.max(img_small)

print(f"İşlenen görüntü boyutu: {img_small.shape}")

# 2. Görüntüden Graph Oluşturma
# Pikselleri düğümlere, parlaklık farklarını kenar ağırlıklarına çevirir.
# Bu fonksiyon sunumdaki W matrisinin temelini oluşturur.
graph = image.img_to_graph(img_small)

# Ağırlıklar, piksel benzerliğine göre (gradient) atanır
# W = e^(-grad / beta) formülüne benzer bir işlem
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# 3. Spektral Kümeleme (N-Cuts Uygulaması)
# eigen_solver='arpack' kullanarak Laplacian matrisinin özdeğerlerini çözer
N_REGIONS = 2 # Resmi kaç parçaya böleceğiz?

print("Spektral kümeleme hesaplanıyor... (Biraz zaman alabilir)")
labels = spectral_clustering(graph, n_clusters=N_REGIONS, eigen_solver='arpack')

# Etiketleri resim boyutuna geri döndür 
segmented_img = labels.reshape(img_small.shape)

# 4. Sonuçları Görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_small, cmap='gray')
ax[0].set_title('Orijinal (Küçültülmüş)')
ax[0].axis('off')

ax[1].imshow(segmented_img, cmap='viridis')
ax[1].set_title(f'Bölütlenmiş Görüntü (N-Cuts, k={N_REGIONS})')
ax[1].axis('off')

plt.tight_layout()
plt.show()