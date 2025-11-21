import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. VERİ SETİ (Senaryo: Tişört Bedeni Tahmini)
# X: [Boy (cm), Kilo (kg)]
raw_data = {
    'Boy':  [150, 160, 165, 155, 170, 180, 185, 175, 190, 178, 162, 182],
    'Kilo': [50,  55,  60,  52,  65,  80,  85,  75,  90,  78,  58,  82],
    'Beden':['S', 'S', 'S', 'S', 'S', 'L', 'L', 'L', 'L', 'L', 'S', 'L']
}

# NumPy dizisine çevirelim
X = np.column_stack((raw_data['Boy'], raw_data['Kilo']))
y = np.array(raw_data['Beden'])

# 2. VERİ ÖLÇEKLEME (STANDARDİZASYON) - ÇOK ÖNEMLİ!
# KNN mesafe (Öklid) hesapladığı için, büyük sayılar (Boy: 170) küçük sayıları (Kilo: 60) ezebilir.
# StandardScaler, tüm verileri eşit şartlara getirir (Ortalama=0, Varyans=1).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. EĞİTİM VE TEST AYRIMI
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 4. MODEL KURULUMU
# n_neighbors=3: En yakın 3 arkadaşına bakacak.
# metric='minkowski', p=2: Standart Öklid mesafesi (kuş uçuşu mesafe).
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# 5. TAHMİN VE SKOR
y_pred = knn_model.predict(X_test)
print(f"Model Doğruluğu: %{accuracy_score(y_test, y_pred) * 100:.2f}")

# 6. GÖRSELLEŞTİRME (Komşuları Görelim)

plt.figure(figsize=(10, 6))

# Eğitim verilerini çiz (Sarı ve Mor noktalar)
# Sınıfları renklere ayırmak için basit bir döngü
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1],
                label=f'Eğitim Verisi ({label})', s=100)

# Yeni bir müşteri geldi diyelim (Boy: 172 cm, Kilo: 70 kg)
yeni_musteri = np.array([[172, 70]])
yeni_musteri_scaled = scaler.transform(yeni_musteri) # Yeni veriyi de mutlaka scale etmeliyiz!

# Modeli kullanarak tahmin et
tahmin = knn_model.predict(yeni_musteri_scaled)
komsular = knn_model.kneighbors(yeni_musteri_scaled, return_distance=False) # En yakınların indeksini al

# Yeni müşteriyi çiz
plt.scatter(yeni_musteri_scaled[:, 0], yeni_musteri_scaled[:, 1],
            color='red', s=200, marker='*', label='Yeni Müşteri (172cm, 70kg)')

# En yakın komşuları daire içine al
for index in komsular[0]:
    plt.scatter(X_train[index, 0], X_train[index, 1],
                s=300, facecolors='none', edgecolors='black', linewidth=2, linestyle='--')

plt.title(f'KNN (K=3): Yeni Müşterinin Tahmini -> {tahmin[0]}')
plt.xlabel('Boy (Standardize Edilmiş)')
plt.ylabel('Kilo (Standardize Edilmiş)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. KULLANICI ROBOTU
print("\n--- Beden Tahmin Robotu ---")
print(f"En yakın {knn_model.n_neighbors} kişiye bakılarak karar veriliyor.")
print(f"Tahmin: {tahmin[0]} Beden (Çünkü en yakın komşuları genelde {tahmin[0]} giyiyor)")