import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #hem eğitip hem test edebileceğimiz model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #hata için


#veriyi hazırlama Evlerin büyüklüğüne bakarak fiyat analizi yapmaca


data= {
    "Ev_Buyuklugu" : [120,250,175,300,220],
    "Fiyat" : [2400000,5000000,3500000,6000000,4400000]
}

df = pd.DataFrame(data) #veriyi DataFrame çevirme

X = df[["Ev_Buyuklugu"]] #girdi DataFrame şeklinde olsun sütunu hepsini adını vs de  aldık bir bütün olark alıyor
y = df["Fiyat"] #çıktı  Seri şeklinde olsun y de sadece sayıları kullancaz analiz için yani başlığı almadık verileri aldık sadece

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42) #yüzde 20 si test verisi olucak eğitilmiş set

#modeli oluşturma

model = LinearRegression() ###doğrusal ilişki modeli var mı
model.fit(X_train,y_train) #eğitilmiş veriler üzerine çalışıyoruz bir nevi eğitiyoruz ve eğitilmiş veriyi modelin için atıyoruz

##############bu kısım kodumuzun düzenli çalışıp çalışmıyacağını test etmek içindi 1 defa yapıp bakıp yorum satırını alman yeterli #######

#hata ne kadar düşükse tahmin o kadar iyidir
##y_pred = model.predict(X_test) #eğitilmiş modeli kullanıcağımızı söylüyoruz X_testi y_predin içine attık
##mse = mean_squared_error(y_test,y_pred)
##rmse = np.sqrt(mse)
###print(rmse)

####################################################################################

ev_buyuklugu =  float(input("Lütfen evin büyüklüğünü giriniz"))
tahmini_fiyat = model.predict([[ev_buyuklugu]])
print(f"Bu evin tahmini fiyatı : {tahmini_fiyat[0]:.2f} TL")
