from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Veriyi Yükle (4 Sütunu var: Yaprak eni, boyu vb.)
iris = load_iris()
X = iris.data
y = iris.target

# 2. PCA Uygula (4 Boyuttan 2 Boyuta düşür)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Görselleştir
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.title(f"PCA ile 4 Boyuttan 2 Boyuta İndirgeme\nBilgi Kaybı: %{100 - sum(pca.explained_variance_ratio_)*100:.2f}")
plt.xlabel('1. Temel Bileşen')
plt.ylabel('2. Temel Bileşen')
plt.show()