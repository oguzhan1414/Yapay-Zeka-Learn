import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error  


df = pd.read_excel("veri_on_isleme_ve_ozellik_muhendisligi.xlsx")


df.fillna(df["Gelir"].mean(),inplace=True) ##boş verileri dolrudduk

#cinsiyet ve meslek sütünlarını sayısal hale getirme çünkü sayısal verilerle işlem yapabiliyoruz bunları çevirmemiz şart

le = LabelEncoder()
df["Cinsiyet"] = le.fit_transform(df["Cinsiyet"])
df["Meslek"] = le.fit_transform(df["Meslek"])

X = df[["Yaş","Meslek"]]
y= df["Gelir"]

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

#modeli test etme 39.10 olarak çıktı ilk olarak
test = model.score(X_test,y_test)
print(f"Modelin Doğruluk oranı : {test*100:.2f}%")

yas = int(input("Yaşınızı giriniz"))
meslek = input("Mesleğini giriniz")

meslek_kod = le.transform([meslek])[0]

yeni_veri = pd.DataFrame([[yas,meslek_kod]],columns=["Yaş","Meslek"])
tahmin = model.predict(yeni_veri)
print(f"Tahmini gelir {tahmin[0]:.2f}")


#Veri ölçeklendirme iki kişinin 80 kilo olduğunu düşün hangisi daha güçlüdür makineye burda ekstra bilgi verip düşüncez