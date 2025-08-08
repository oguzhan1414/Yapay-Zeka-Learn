import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.models import Sequential
from keras.layers import Dense

#giriş verileri (XOR problemi olarak al)

X = np.array([[0,0],[0,1],[1,0],[1,1]]) # bu problem  0 0 xor 0 olur 0 1 xor 1 oluyor gibi   input
y = np.array([[0],[1],[1],[0]]) # bu da çözümü    output

#basit model
model = Sequential()
model.add(Dense(3,input_dim=2,activation='relu')) ##problemi çözerken 4 tane nöron kullanıyor Dense(4) 4 farklı çözüm yolu aramak gibi bir şeydir
model.add(Dense(1,activation='sigmoid'))

#modeli derleme
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#modeli eğitme
model.fit(X,y,epochs=200,verbose=1)

tahminleme = model.predict(X)
print("Tahminler: \n",tahminleme)