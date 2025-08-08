import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

df = pd.read_excel("kredi_onay_verisi_1000.xlsx")
X = df[["Yaş","Gelir"]].values
y = df["Kredi Onayı"].values

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#smote ile veri dengesini sağlayalım

smote = SMOTE(random_state=42)
X_train_smote , y_train_smote = smote.fit_resample(X_train,y_train)


##veri örçeklendirme
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

##modeli oluşturma
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote_scaled,y_train_smote)

#tahminleme
y_pred = model.predict(X_test_scaled)
accuarcy = accuracy_score(y_test,y_pred)
print(f"Test Doğrluk oranı : {accuarcy*100:.2f}")

####yapay sinir ağları ile doğruluk oranı arttırmaca
nn_model = Sequential()
nn_model.add(Dense(2,input_dim=2,activation="relu"))
nn_model.add(Dense(1,activation="sigmoid"))

#modeli derle
optimizer = Adam(learning_rate=0.001)
nn_model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])

#modeli eğitelim
nn_model.fit(X_train_smote_scaled,y_train_smote,epochs=75,batch_size=32,verbose=1)

#yapay sinir ağı tahminleme ve doğruluk oranı ölçümü
y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype("int32")
nn_accuracy = accuracy_score(y_test,y_pred_nn)
print(f"yapay sinir ağı test doğruluk oranı: {nn_accuracy *100:.2f}")

