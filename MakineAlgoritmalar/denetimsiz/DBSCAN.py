import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 1. Zorlu Bir Veri Seti Oluşturalım (Hilal Şeklinde)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 2. Modeli Kuralım
# eps: Komşuluk yarıçapı (Ne kadar yakınsa komşu sayılır?)
# min_samples: Bir yerin "küme" sayılması için en az kaç nokta olmalı?
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# 3. Görselleştirme
plt.figure(figsize=(8, 5))
# labels == -1 olanlar gürültüdür (Noise)
unique_labels = set(labels)
colors = ['blue', 'red', 'green', 'purple']

for label in unique_labels:
    if label == -1:
        # Gürültü noktalarını siyah ve küçük çiz
        plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c='black', s=20, label='Gürültü (Noise)')
    else:
        plt.scatter(X[labels == label, 0], X[labels == label, 1], c=colors[label], s=60, label=f'Küme {label}')

plt.title("DBSCAN Kümeleme (K-Means bunu yapamazdı)")
plt.legend()
plt.show()