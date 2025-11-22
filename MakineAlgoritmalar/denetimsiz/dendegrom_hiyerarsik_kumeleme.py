import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# 1. VERİ SETİ (Matematik Notu, Fizik Notu)
# Veriyi dağınık gruplar oluşturacak şekilde elle hazırlıyorum.
data = {
    'Ogrenci_ID': ['Ali', 'Ayse', 'Mehmet', 'Fatma', 'Can', 'Ece', 'Cem'],
    'Matematik':  [10,   12,     85,       88,      45,    48,    82],
    'Fizik':      [15,   18,     82,       90,      50,    52,    85]
}

df = pd.DataFrame(data)
X = df[['Matematik', 'Fizik']].values # Sadece notları alıyoruz

plt.figure(figsize=(12, 5))

# 2. DENDROGRAM ÇİZİMİ (Karar Aşaması)
# Bu grafik bize hangi noktaların birbirine daha yakın olduğunu gösterir.
plt.subplot(1, 2, 1)
plt.title("Dendrogram (Ağaç Grafiği)")
plt.xlabel("Öğrenciler (İndeks)")
plt.ylabel("Öklid Mesafesi (Uzaklık)")

# 'ward' metodu, kümelerin varyansını minimize ederek birleştirir.
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# 3. MODELİ EĞİTME (Agglomerative Clustering)
# Dendrograma baktık ve 3 ana dal olduğunu gördük diyelim (K=3 seçiyoruz).
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_hc = hc.fit_predict(X) # Kümeleri tahmin et

# 4. SONUÇLARI GÖRSELLEŞTİRME (Scatter Plot)
plt.subplot(1, 2, 2)
plt.title("Hiyerarşik Kümeleme Sonucu")
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Küme 1 (Başarılı)')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Küme 2 (Zayıf)')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Küme 3 (Orta)')

plt.xlabel('Matematik Notu')
plt.ylabel('Fizik Notu')
plt.legend()

plt.tight_layout()
plt.show()

# Hangi öğrenci hangi grupta?
df['Grup_No'] = y_hc
print(df)