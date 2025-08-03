import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

##veriyi hazırlama : yaş ,kan basaıncı , kolestrol ve hastalık durumu verisi


"""
data = {
    "Yas" : [25,50,45,30,60],
    "Kan_Basinci": [120,140,130,110,150],
    "Kolestrol" : [180,240,200,160,220],
    "Hastalik" : [0,1,1,0,1] #0 hayır , 1 evet
}

df = pd.DataFrame(data)
"""

df = pd.read_excel("karar_agaci_veri_100.xlsx")
X = df[["Yas","Kan_Basinci","Kolesterol"]]
y = df["Hastalik"]

X_train , X_test , y_train,y_test = train_test_split(X,y ,test_size=0.2,random_state=42)

classiefier = DecisionTreeClassifier() #karar ağacları üstte hem eğittik hem test ettik
classiefier.fit(X_train,y_train)

##y_pred = classiefier.predict(X_test)
##accuracy = accuracy_score(y_test,y_pred)
##print(accuracy)

yas = int(input("Yaşını giriniz"))
kan_basinci = int(input("Kan basinci gir"))
kolestrol = int(input("Kolestrol seviyenizi girin"))

##tahmin oluşturma
tahmin = classiefier.predict([[yas,kan_basinci,kolestrol]])
sonuc = "Hastalık Var" if tahmin == 1 else "Hastalık Yok"
print(sonuc)