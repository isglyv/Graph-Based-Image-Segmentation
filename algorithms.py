import time
import numpy as np
from sklearn.cluster import KMeans
from skimage import segmentation, color, graph
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


class SegmentationBenchmarker:
    def __init__(self, image_data):
        self.img = image_data
        self.results = {}
        self.timings = {}

    def run_kmeans(self, n_clusters=3):
        """
        Klasik K-Means Kümeleme (Rakip Algoritma)
        Sadece piksel renklerine bakar, komşuluk ilişkisini bilmez.
        """
        print(f"--- K-Means (k={n_clusters}) Çalıştırılıyor ---")
        start_time = time.time()

        # Resmi düzleştir (piksel listesine çevir)
        X = self.img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Etiketleri resim boyutuna geri getir
        segmented_img = labels.reshape(self.img.shape[:2])
        # Renklendir
        out_img = color.label2rgb(segmented_img, self.img, kind='avg', bg_label=0)

        duration = time.time() - start_time
        self.results['K-Means'] = out_img
        self.timings['K-Means'] = duration
        print(f"Tamamlandı. Süre: {duration:.4f} sn")
        return out_img

    def run_spectral_slic(self, n_segments=300, compactness=20, n_clusters=3):
        """
        Graph-Based Segmentation (Bizim Yöntem - SLIC + Spectral Clustering)
        Karmaşık Ağ Teorisi kullanır. Resmi zorla 'n_clusters' kadar parçaya böler.
        """
        print(f"--- Graph-Based Spectral (Segments={n_segments}) Çalıştırılıyor ---")
        start_time = time.time()

        # 1. Adım: SLIC ile Süper Pikseller (Düğüm Sayısını Azaltma)
        # Bu işlem resmi küçük mozaiklere böler
        labels = segmentation.slic(self.img, compactness=compactness, n_segments=n_segments, start_label=0)

        # 2. Adım: Region Adjacency Graph  Oluşturma
        # Düğümler = Süper Pikseller, Kenarlar = Renk Benzerliği
        g = graph.rag_mean_color(self.img, labels)

        # 3. Adım: Spektral Kümeleme
        # Grafiğin komşuluk matrisini (affinity matrix) alıyoruz
        # Bu matris, hangi parçaların birbirine benzediğini matematiksel olarak tutar.
        import networkx as nx
        affinity_matrix = nx.to_scipy_sparse_array(g, format='csr')

        affinity_matrix.indices = affinity_matrix.indices.astype(np.int32)
        affinity_matrix.indptr = affinity_matrix.indptr.astype(np.int32)

        # Matrisi Spektral Kümeleme algoritmasına sokuyoruz
        # assign_labels='discretize' genellikle görüntü işlemede daha kararlı sonuç verir
        sc = spectral_clustering(affinity_matrix, n_clusters=n_clusters, eigen_solver='arpack',
                                 assign_labels='discretize', random_state=42)

        # Spektral kümeleme bize süper piksellerin hangi gruba ait olduğunu verir (0, 1, 2...)
        # Şimdi bu etiketleri orijinal resme geri yansıtmamız lazım.

        # Yeni etiket haritası oluştur
        labels_new = labels.copy()
        for i, cluster_label in enumerate(sc):
            labels_new[labels == i] = cluster_label

        # Sonucu görselleştir
        out_img = color.label2rgb(labels_new, self.img, kind='avg', bg_label=-1)

        duration = time.time() - start_time
        self.results['Graph-Based'] = out_img
        self.timings['Graph-Based'] = duration
        print(f"Tamamlandı. Süre: {duration:.4f} sn")
        return out_img