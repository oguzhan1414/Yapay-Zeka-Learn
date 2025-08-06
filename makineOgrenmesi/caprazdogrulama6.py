import pandas as pd
from sklearn.model_selection import cross_val_score #çaprazlama bir doğrulama kullaanmak istiyoruz
from sklearn.tree import DecisionTreeClassifier ##veriler üzerine kaategorize etmek için tree kullanır evet hayır kategorize ettiğimizi düşün
from sklearn.metrics import accuracy_score
#capraz doğrulama hem az hem fazla verilerde kullanılır random_state ise daha çok az verilerde kullanılır

df = pd.read_excel("karar_agaci_veri_100.xlsx")

x = df[['Yas','Kan_Basinci','Kolesterol']]
y = df['Hastalik']

classifier = DecisionTreeClassifier(max_depth=1,min_samples_leaf=3,min_samples_split=2) #eğitilmiş veri
#cross_val_skorlar = cross_val_score(classifier,x,y,cv=5) #verimizi test etmek için kullanırız 5 kere yapıyoruz burdaa
classifier.fit(x,y)
#print(cross_val_skolar)
#print(cross_val_skolar.mean())

yas = int(input("Yaşı Gir: "))
kan_basinci = float(input("Kan Basıncı: "))
kolesterol = float((input("kolesterol: ")))

#kullanıcıdan alaınan veriyi modelin anlayacağı formata çevirme
yeni_veri = pd.DataFrame([[yas,kan_basinci,kolesterol]],columns=['Yas','Kan_Basinci','Kolesterol'])

tahmin = classifier.predict(yeni_veri)
if tahmin[0] ==1 :
    print("Tahmin: Hastalık Var")
else:
    print("Tahmin: Hastalık Yok")