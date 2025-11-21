from sklearn.linear_model import LinearRegression
from  sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. VERİ HAZIRLIĞI
# X: Metrekareler (Bağımsız Değişken)  y: Fiyatlar (Bağımlı Değişken)
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350]])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000])
##veileri eğitim ve test ayırma
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 3. MODEL EĞİTİMİ
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print(f"--- Model Performansı ---")
print(f"Ortalama Hata Karesi (MSE): {mse:.2f}")
print(f"R2 Skoru (Başarı Oranı): {r2:.2f}")
print(f"Eğim (Coefficient): {model.coef_[0]:.2f}")
print(f"Sabit (Intercept): {model.intercept_:.2f}")
plt.scatter(X_train, y_train, color='blue', label='Eğitim Verisi')
plt.scatter(X_test, y_test, color='red', label='Test Verisi')
plt.plot(X, model.predict(X), color='green', linewidth=2, label='Tahmin Çizgisi')
plt.xlabel('Metrekare')
plt.ylabel('Fiyat')
plt.title('Ev Fiyat Tahmin Modeli')
plt.legend()
plt.show()