import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. VERİ SETİ OLUŞTURMA (Senaryo: Çalışma Saati vs. Sınav Sonucu)
# X: Çalışma Saatleri (Bağımsız Değişken)
# Daha gerçekçi olması için biraz karışık veri üretiyoruz
X = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5,
              2.75, 3.0, 3.25, 3.50, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5]).reshape(-1, 1)

# y: Sonuç (0 = Kaldı, 1 = Geçti) (Bağımlı Değişken)
# Az çalışanlar genelde kalmış (0), çok çalışanlar geçmiş (1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
              1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# 2. VERİ AYIRMA (Eğitim ve Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL KURULUMU VE EĞİTİMİ
# Lojistik regresyon 'solver' parametresi optimizasyon için kullanılır, küçük veride 'liblinear' iyidir.
log_model = LogisticRegression(solver='liblinear')
log_model.fit(X_train, y_train)

# 4. TAHMİN VE DEĞERLENDİRME
y_pred = log_model.predict(X_test)

# Başarı Oranı (Accuracy)
acc = accuracy_score(y_test, y_pred)
# Karmaşıklık Matrisi (Hangi sınıfları karıştırdı?)
cm = confusion_matrix(y_test, y_pred)

print(f"--- Model Sonuçları ---")
print(f"Doğruluk Oranı (Accuracy): %{acc * 100:.2f}")
print(f"Confusion Matrix:\n{cm}")
print("\n--- Detaylı Rapor ---")
print(classification_report(y_test, y_pred))

# 5. GÖRSELLEŞTİRME (Sigmoid Eğrisi)




plt.figure(figsize=(10, 6))

# Gerçek verileri nokta olarak çiz
plt.scatter(X, y, color='red', label='Gerçek Veriler (0:Kaldı, 1:Geçti)')

# Sigmoid Fonksiyonunu (S Eğrisini) çizmek için aralık oluşturma
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = log_model.predict_proba(X_range)[:, 1]  # 1 olma olasılığını alıyoruz

plt.plot(X_range, y_prob, color='blue', linewidth=2, label='Lojistik Regresyon Eğrisi (Olasılık)')
plt.axhline(0.5, color='gray', linestyle='--', label='Karar Sınırı (0.5)')  # %50 olasılık çizgisi

plt.xlabel('Çalışma Saati')
plt.ylabel('Geçme Olasılığı (0-1 Arası)')
plt.title('Çalışma Saati ve Sınav Başarısı Analizi')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. KULLANICI ETKİLEŞİMİ (Olasılık Tahmini)
print("\n--- Sınav Sonucu Tahmin Robotu ---")
while True:
    try:
        user_input = input("Kaç saat ders çalıştınız? (Çıkış için 'q'): ")
        if user_input.lower() == 'q':
            break

        hours = float(user_input)

        # Sadece 0 veya 1 demek yerine olasılığı da verelim
        tahmin_sinifi = log_model.predict([[hours]])[0]
        tahmin_olasiligi = log_model.predict_proba([[hours]])[0][1]  # Geçme olasılığı

        sonuc_metni = "GEÇER" if tahmin_sinifi == 1 else "KALIR"

        print(f"📚 {hours} saat çalışma ile tahmin: **{sonuc_metni}**")
        print(f"📊 Geçme İhtimali: %{tahmin_olasiligi * 100:.2f}\n")

    except ValueError:
        print("Lütfen sayısal bir değer girin!")