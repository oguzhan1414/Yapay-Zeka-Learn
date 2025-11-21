import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 1. VERİ SETİ (Senaryo: Tümör Büyüklüğü ve Yoğunluğu)
# make_blobs: Rastgele kümelenmiş veri üretir (Eğitim amaçlı idealdir)
# centers=2: İki farklı sınıf olsun (İyi huylu / Kötü huylu)
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.2)

# X: [Tümör Boyutu, Tümör Yoğunluğu]
# y: 0 (İyi Huylu), 1 (Kötü Huylu)

# 2. EĞİTİM VE TEST AYRIMI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL KURULUMU
# kernel='linear': Verileri düz bir çizgiyle ayırır.
# C=1.0: Hata toleransı. Düşük C daha geniş marjin (daha toleranslı), Yüksek C daha katı ayrım demektir.
clf = svm.SVC(kernel='linear', C=1.0) 
clf.fit(X_train, y_train)

# 4. TAHMİN
y_pred = clf.predict(X_test)
print(f"Model Doğruluğu: %{accuracy_score(y_test, y_pred) * 100:.2f}")

# 5. GÖRSELLEŞTİRME (SVM'in İmzası: Decision Boundary ve Marjinler)

plt.figure(figsize=(10, 6))

# Veri noktalarını çiz
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', edgecolors='k', label='Veriler')

# Karar sınırlarını ve marjinleri çiz (Scikit-learn'in yeni görselleştirme aracı)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1], # -1: Alt sınır, 0: Karar çizgisi, 1: Üst sınır
    alpha=0.5,
    linestyles=["--", "-", "--"], # Marjinler kesikli, orta çizgi düz
    ax=plt.gca()
)

# Destek Vektörlerini (Support Vectors) İşaretle
# Modelin karar verirken "baz aldığı" kritik noktalar bunlardır.
sv = clf.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=150, linewidth=2, facecolors='none', edgecolors='r', label='Destek Vektörleri')

plt.title('SVM: Tümör Sınıflandırması (Kırmızı Halkalar = Destek Vektörleri)')
plt.xlabel('Tümör Boyutu')
plt.ylabel('Tümör Yoğunluğu')
plt.legend(loc="upper right")
plt.show()

# 6. KULLANICI TAHMİNİ
print("\n--- Tıbbi Teşhis Asistanı ---")
sample_tumor = [[7.5, -8.0]] # Örnek bir tümör verisi
sonuc = clf.predict(sample_tumor)

print(f"Yeni Hasta Verisi: {sample_tumor}")
if sonuc[0] == 0:
    print("Teşhis: 🟢 İYİ HUYLU (Benign)")
else:
    print("Teşhis: 🔴 KÖTÜ HUYLU (Malignant)")