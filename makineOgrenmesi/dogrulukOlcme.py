# Gerekli kütüphaneleri içe aktar
import pandas as pd
from sklearn.model_selection import train_test_split  # Veriyi eğitim/test olarak böler
from sklearn.tree import DecisionTreeClassifier       # Karar ağacı sınıflandırıcısı
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Değerlendirme metrikleri
import matplotlib.pyplot as plt  # Grafik çizimi için

# Excel dosyasını oku (veri kümesini içe aktar)
df = pd.read_excel("karar_agaci_veri_100.xlsx")

# Girdi (özellik) ve çıktı (hedef) sütunlarını ayır
X = df[["Yas", "Kan_Basinci", "Kolesterol"]]  # Bağımsız değişkenler
y = df["Hastalik"]                            # Bağımlı değişken (etiket)

# Veriyi eğitim ve test seti olarak ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı sınıflandırıcısı oluştur
classiefier = DecisionTreeClassifier()

# Modeli eğitim verisiyle eğit
classiefier.fit(X_train, y_train)

# Test verisiyle tahmin yap
y_pred = classiefier.predict(X_test)

# Karışıklık matrisi oluştur (gerçek vs tahmin karşılaştırması)
cm = confusion_matrix(y_test, y_pred)

# Karışıklık matrisini görselleştir
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Başlık ekleyip matrisi ekranda göster
plt.title("Matrix - Karar Ağacı")
plt.show()
