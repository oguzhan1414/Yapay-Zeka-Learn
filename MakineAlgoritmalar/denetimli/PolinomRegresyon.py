import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. SENARYO: Maaş ve Tecrübe İlişkisi (Doğrusal olmayan veri)
# Üst düzey pozisyonlarda maaşlar katlanarak artıyor.
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) # Tecrübe Yılı
y = np.array([4500, 5000, 6000, 8000, 11000, 15000, 20000, 30000, 50000, 100000]) # Maaş

# 2. POLİNOM DÖNÜŞÜMÜ (Sihirli Kısım)
# degree=3 veya 4 seçelim (Maaş artışı çok dik olduğu için)
degree = 3
poly_features = PolynomialFeatures(degree=degree)

# X verisini alıp [x, x^2, x^3] haline getiriyor
X_poly = poly_features.fit_transform(X)

# 3. MODEL EĞİTİMİ
# Dönüştürülmüş (X_poly) veriyi Linear Regression'a sokuyoruz
model = LinearRegression()
model.fit(X_poly, y)

# 4. GÖRSELLEŞTİRME
plt.figure(figsize=(10, 6))

# Gerçek veriler
plt.scatter(X, y, color='red', label='Gerçek Maaşlar')

# Tahmin eğrisini pürüzsüz çizmek için ara noktalar oluşturma
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
X_grid_poly = poly_features.transform(X_grid)
y_pred_grid = model.predict(X_grid_poly)

plt.plot(X_grid, y_pred_grid, color='blue', linewidth=2, label=f'Polinom Model (Derece {degree})')

plt.title(f'Tecrübe vs Maaş (Polinom Derecesi: {degree})')
plt.xlabel('Tecrübe (Yıl)')
plt.ylabel('Maaş (TL)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. TAHMİN YAPALIM
yil = 4.5
# Tahmin yaparken de önce veriyi dönüştürmeliyiz!
tahmin_poly = poly_features.transform([[yil]])
sonuc = model.predict(tahmin_poly)

print(f"{yil} yıllık tecrübe için tahmin edilen maaş: {sonuc[0]:.2f} TL")
print(f"Model Başarısı (R2): {r2_score(y, model.predict(X_poly)):.4f}")