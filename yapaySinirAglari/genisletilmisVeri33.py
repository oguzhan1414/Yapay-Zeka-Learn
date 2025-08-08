import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_excel('genisletilmisveri.xlsx')



#pandas veri çerçevesi
df = pd.DataFrame(data)
#sahte değişkenler oluştur

df = pd.get_dummies(df , columns=['Medeni Durum', 'Meslek','Eğitim Durumu'] , drop_first=True)

#kesme işlemi ya da çıkartma işlemi
X = df.drop('Kredi Onayı', axis=1).values
#Çıkış Verisis
y = df['Kredi Onayı'].values
#veriyi ikiye böl  /test ve / eğitim olarak

X_train , X_Test , y_train, y_test = train_test_split(X,y, test_size=0.4 , random_state=42)
#Modeli oluştur

model = Sequential()
model.add(Dense(3 , input_dim=X.shape[1] , activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
optimizer = Adam(learning_rate=0.002)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#veriyi ölçeklendirelim

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_Test_scaled = scaler.transform(X_Test)

#modeli eğitilem
model.fit(X_train_scaled, y_train, epochs=350, verbose=1)

#tahminleme
y_pred = model.predict(X_Test_scaled)
y_pred = (y_pred > 0.5).astype(int)

#doğruluk oranını ölç
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Doğruluk Oranı : {accuracy * 100:.2f}%')

#kullanıcıdan veri alma

while True:
    user_input_1 = float(input('Yaşınızı Giriniz : '))
    user_input_2 = float(input('Maaşınızı Giriniz : '))
    user_input_3 = input('Medeni Durumunuzu Giriniz(Bekar, Evli):')
    user_input_4 = input('Mesleğiniz (Mühendis, Doktor, Öğretmen, Avukat):')
    user_input_5 = input('Eğitim Durumunuzu Giriniz (Lisans, Yüksek Lisans, Doktora):')

    user_data = pd.DataFrame({
        'Yaş': [user_input_1],
        'Gelir': [user_input_2],
        'Medeni Durum_' + user_input_3:[1],
        'Meslek_' + user_input_4:[1],
        'Eğitim Durumu_' + user_input_5:[1]
    })

    user_Data = user_data.reindex(columns=df.drop('Kredi Onayı', axis=1).columns , fill_value=0)

    #veriyi ölçeklendirelim

    user_data_scaled = scaler.transform(user_Data)

    #tahminleme
    prediction = model.predict(user_data_scaled)
    print(f'Tahmin Sonucu :{prediction[0][0]:.4f}')
