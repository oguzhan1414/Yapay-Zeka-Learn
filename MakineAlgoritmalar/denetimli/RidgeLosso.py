import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 1. VERİ OLUŞTURMA
# 10 tane özelliğimiz (feature) var ama aslında sonucu belirleyen sadece 3 tanesi!
# Diğer 7 özellik gürültü (noise).
X, y, coef = make_regression(n_samples=100, n_features=10, n_informative=3,
                             noise=10, coef=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. MODELLERİ KURMA
# Linear Regression (Standart)
lr = LinearRegression()
lr.fit(X_train, y_train)

# Ridge Regression (L2 Regularization)
# alpha: Ceza katsayısı. Ne kadar büyükse o kadar çok baskılar.
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression (L1 Regularization)
# alpha: Ceza katsayısı.
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 3. SONUÇLARI KIYASLAMA (Skorlar)
print(f"Linear Train Score: {lr.score(X_train, y_train):.3f} | Test Score: {lr.score(X_test, y_test):.3f}")
print(f"Ridge  Train Score: {ridge.score(X_train, y_train):.3f} | Test Score: {ridge.score(X_test, y_test):.3f}")
print(f"Lasso  Train Score: {lasso.score(X_train, y_train):.3f} | Test Score: {lasso.score(X_test, y_test):.3f}")

# 4. GÖRSELLEŞTİRME (Katsayıların Durumu)
plt.figure(figsize=(12, 6))

plt.plot(lr.coef_, 's', label='Linear Regression', markersize=10, alpha=0.7)
plt.plot(ridge.coef_, '^', label='Ridge Regression (alpha=1)', markersize=10, alpha=0.7)
plt.plot(lasso.coef_, 'o', label='Lasso Regression (alpha=0.1)', markersize=10, alpha=0.9)

plt.xlabel('Özellik İndeksi (0-9 arası)')
plt.ylabel('Katsayı Değeri (Coefficient)')
plt.hlines(0, 0, 9, color='black', linestyle='--', linewidth=2) # 0 çizgisi
plt.title('Linear vs Ridge vs Lasso: Katsayı Karşılaştırması')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. LASSO'NUN GÜCÜ (Hangi özellikleri sildi?)
print("\n--- Lasso Hangi Özellikleri Sıfırladı? ---")
for i, katsayi in enumerate(lasso.coef_):
    if katsayi == 0:
        print(f"Özellik {i}: SIFIRLANDI (Gereksiz görüldü)")
    else:
        print(f"Özellik {i}: Kullanıldı ({katsayi:.2f})")