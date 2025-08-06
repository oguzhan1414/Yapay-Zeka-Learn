#Veri ölçeklendirme iki kişinin 80 kilo olduğunu düşün hangisi daha güçlüdür makineye burda ekstra bilgi verip düşüncez
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_excel("veri_on_isleme_ve_ozellik_muhendisligi.xlsx")

df.fillna(df["Gelir"].mean(),inplace=True) ##boş verileri dolrudduk


le = LabelEncoder()
df["Cinsiyet"] = le.fit_transform(df["Cinsiyet"])
df["Meslek"] = le.fit_transform(df["Meslek"])

X = df[["Yaş","Meslek","Cinsiyet"]]
y= df["Gelir"]


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  #fit eğit transfrom çevir demektir eğitilmiş verileri standarda çevirme
X_test = scaler.transform(X_test)

#modeli oluşturma
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

###Cinsiyet ekliyerek doğruluk oranı 71.38 e kadar çıktı
test = linear_model.score(X_test,y_test)
print(f"DOĞRULUK {test*100:.2f}" )

###daha karmaşık bir model kullan
rf_model = RandomForestRegressor(n_estimators=100,random_state=42) ##100 karar ağacı oluşturma
rf_model.fit(X_train,y_train)
rf_test = rf_model.score(X_test,y_test)
print(f"DOĞRULUK {rf_test*100:.2f}" )

yas = int(input("Yaşınızı giriniz"))
meslek = input("Mesleğini giriniz")
cinsiyet = input("Cinsiyet gir")

if cinsiyet == "Erkek":
    cinsiyet_kod = 0
elif cinsiyet == "Kadın":
    cinsiyet_kod = 1
else:
    raise ValueError("Geçersiz Giriş")

meslek_kod = le.transform([meslek])[0]
yeni_veri = pd.DataFrame([[yas,meslek_kod,cinsiyet_kod]],columns=["Yaş","Meslek","Cinsiyet"])
yeni_veri_sclaed = scaler.transform(yeni_veri)
tahmin = rf_model.predict(yeni_veri_sclaed)
print(f"Tahmini gelir {tahmin[0]:.2f}")




