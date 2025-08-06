import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_excel("veri_on_isleme_ve_ozellik_muhendisligi.xlsx")

df.fillna(df["Gelir"].mean(),inplace=True)

le = LabelEncoder()
df["Cinsiyet"] = le.fit_transform(df["Cinsiyet"])
df["Meslek"] = le.fit_transform(df["Meslek"])

X = df[["Yaş","Meslek","Cinsiyet"]]
y = df["Gelir"]

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

###ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#modeli oluştur ve çapraz olarak değerlendir
model = LinearRegression()
cv_score = cross_val_score(model,X_train,y_train,cv=5,scoring="neg_mean_squared_error")
cv_rmse_scroe = (-cv_score) ** 0.5
print(f"Liner Regresyon 5 Katmanlı Cross Val Puanı : {cv_rmse_scroe.mean():.2f}")


#random forest karar ağacı ile çapraz doğrulama performansını ölçelim
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
cv_score_rf = cross_val_score(rf_model,X_train,y_train,cv=5,scoring="neg_mean_squared_error")
cv_rmse_score_rf =(-cv_score_rf) ** 0.5
print(f"Random Fores 5 Katmanlı Cross Val Puanı : {cv_rmse_score_rf.mean():.2f}")