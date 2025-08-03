import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #hem eğitip hem test edebileceğimiz model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #hata için


data= {
    "Ev_Buyuklugu" : [120,250,175,300,220],
    "Oda_Sayisi" : [3,5,4,6,4],
    "Fiyat" : [2400000,5000000,3500000,6000000,4400000]
}

df = pd.DataFrame(data)

X = df[["Ev_Buyuklugu","Oda_Sayisi"]]
y = df["Fiyat"]

#modeli eğitme
X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

ev_buyuklu = float(input("Lütfen evin büyüklüğünü giriniz: "))
oda_sayisi = int(input("Lütfen oda sayisini giriniz: "))
tahmini_fiyat = model.predict([[ev_buyuklu,oda_sayisi]]) #veriyi eğitme eğitilmiş veriye kullanıcın girdiği veriye göre bakıcak
print(f"Tahmini Fiyat : {tahmini_fiyat[0]:.0f}")
