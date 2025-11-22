import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. VERİ SETİNİ OLUŞTURALIM
# [Yıllık Gelir (k TL), Harcama Puanı (1-100)]
# Veriyi manuel oluşturuyorum ki grupları net görebilesin.
data = {
    'Gelir': [15, 16, 17, 80, 85, 88, 45, 50, 48, 19, 82, 49],
    'Puan':  [80, 85, 82, 15, 20, 18, 50, 45, 55, 78, 16, 52]
}

df = pd.DataFrame(data)

# 2. MODELİ KURMA VE EĞİTME
# K=3 seçiyoruz (3 grup oluşturacağız)
kmeans = KMeans(n_clusters=3, random_state=42)

# Modeli veriye uydur (Fit) ve tahmin et (Predict)
# Algoritma burada merkezleri buluyor ve noktaları atıyor.
df['Kume_No'] = kmeans.fit_predict(df[['Gelir', 'Puan']])

# Merkez noktalarının (Centroids) koordinatlarını alalım
merkezler = kmeans.cluster_centers_

# 3. SONUÇLARI GÖRSELLEŞTİRME
plt.figure(figsize=(8, 6))

# Her kümeyi farklı renkte çizelim
plt.scatter(df[df['Kume_No'] == 0]['Gelir'], df[df['Kume_No'] == 0]['Puan'],
            s=100, c='red', label='Küme 1')
plt.scatter(df[df['Kume_No'] == 1]['Gelir'], df[df['Kume_No'] == 1]['Puan'],
            s=100, c='blue', label='Küme 2')
plt.scatter(df[df['Kume_No'] == 2]['Gelir'], df[df['Kume_No'] == 2]['Puan'],
            s=100, c='green', label='Küme 3')

# Merkez noktalarını (Centroids) sarı yıldız olarak gösterelim
plt.scatter(merkezler[:, 0], merkezler[:, 1], s=300, c='yellow', marker='*', label='Merkezler')

plt.title('Müşteri Segmentasyonu (K-Means)')
plt.xlabel('Yıllık Gelir (Bin TL)')
plt.ylabel('Harcama Puanı (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Sonuçları yazdıralım
print("Veri Seti ve Atanan Kümeler:\n")
print(df)