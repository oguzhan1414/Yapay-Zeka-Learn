import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

#giriş verileri (Yaş ve Gelir)

X = np.array([[25,2000],[30,4000],[45,10000],[50,3000]]) # bu problem  0 0 xor 0 olur 0 1 xor 1 oluyor gibi   input
y = np.array([[0],[1],[1],[0]]) # bu da çözümü    output

#basit model
model = Sequential()
model.add(Dense(6,input_dim=2,activation='relu')) ##problemi çözerken 4 tane nöron kullanıyor Dense(4) 4 farklı çözüm yolu aramak gibi bir şeydir
model.add(Dense(1,activation='sigmoid'))

#modeli derleme
optimizer = Adam(learning_rate=0.005) ##öğrenme hızı bu şekilde ayarladığımızda daha iyi sonuç verdi
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

#Veriyi Ölçeklendir  ölçeklendirdiğimizde çok daha doğru sonuçlar aldık
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X) #ortalaması 0 standart sapmasaı 1 olacak şekilde eğit
#modeli eğitme
model.fit(X_scaler,y,epochs=200,verbose=1)

tahminleme = model.predict(X_scaler)
print("Tahminler: \n",tahminleme)

while True:
    user_input_1 = float(input("Yaşınızı Girin"))
    user_input_2 = float(input("Maaşınızı Girin"))

    #kullanıcıdan alınan verileri modele uygun hale getirme
    user_data = np.array([[user_input_1,user_input_2]])
    #aldığımız verileri ölçeklendirme
    user_data_sclaed = scaler.transform(user_data)
    #tahmin oluştur
    prediction = model.predict(user_data_sclaed)
    if(prediction>0.50):
        print("Kredi Alabilirsin")
    else:
        print("Kredi alamazsın")