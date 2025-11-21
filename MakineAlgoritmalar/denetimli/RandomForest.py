import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. VERİ SETİ OLUŞTURMA (Senaryo: Banka Kredi Başvuruları)
# Özellikler: [Yıllık Gelir (bin TL), Kredi Skoru (0-800), Borç Miktarı (bin TL), Yaş]
data = {
    'Gelir': [50, 120, 40, 300, 80, 25, 60, 200, 45, 90, 300, 35, 150, 55, 220],
    'Kredi_Skoru': [600, 750, 550, 780, 680, 400, 620, 790, 580, 710, 800, 500, 760, 610, 740],
    'Borc': [10, 5, 20, 10, 15, 25, 12, 5, 30, 8, 2, 18, 10, 14, 8],
    'Yas': [25, 40, 22, 50, 30, 21, 28, 45, 24, 35, 55, 23, 42, 29, 48],
    'Onay': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 0:Red, 1:Onay
}

df = pd.DataFrame(data)

X = df[['Gelir', 'Kredi_Skoru', 'Borc', 'Yas']]
y = df['Onay']

# 2. EĞİTİM VE TEST AYRIMI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. MODEL KURULUMU (Ormanı İnşa Etme)
# n_estimators=100 -> Bu ormanda 100 tane karar ağacı olacak.
# random_state=42 -> Sonuçlar her çalıştırmada aynı olsun diye.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. TAHMİN VE DEĞERLENDİRME
y_pred = rf_model.predict(X_test)

print(f"Model Doğruluğu: %{accuracy_score(y_test, y_pred) * 100:.2f}")
print("\n--- Detaylı Rapor ---")
print(classification_report(y_test, y_pred))

# 5. ÖZELLİK ÖNEM DÜZEYİ (Feature Importance) - EN ÖNEMLİ KISIM
# Random Forest'ın en sevilen yanı: Hangi kriterin kararda daha etkili olduğunu söyler.
onem_dereceleri = rf_model.feature_importances_
ozellik_isimleri = X.columns

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x=onem_dereceleri, y=ozellik_isimleri, palette='viridis')
plt.title('Kredi Onayında Hangi Özellik Ne Kadar Önemli?')
plt.xlabel('Önem Derecesi (0-1 arası)')
plt.ylabel('Özellikler')
plt.show()

# 6. KULLANICI İLE TAHMİN
print("\n--- Banka Kredi Robotu ---")
try:
    # Örnek: 65.000 TL Gelir, 630 Skor, 12.000 TL Borç, 27 Yaş
    print("Örnek Müşteri Analiz Ediliyor...")
    ornek_musteri = [[65, 630, 12, 27]]

    sonuc = rf_model.predict(ornek_musteri)

    if sonuc[0] == 1:
        print("Sonuç: ✅ KREDİ ONAYLANDI")
    else:
        print("Sonuç: ❌ KREDİ REDDEDİLDİ")

except Exception as e:
    print("Hata:", e)