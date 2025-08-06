import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("Kategorik_Gelir.xlsx")

####eksik verileri doldurma

df["Gelir"].fillna(df["Gelir"].mean(),inplace=True)

le = LabelEncoder()
df["Cinsiyet"] = le.fit_transform(df["Cinsiyet"]) ##kadın erkeği 0 ve 1 e çevirme

scaler = StandardScaler() ##verimizi standart hale getirmeye yarıyor

#df[["Yaş","Gelir"]] = scaler.fit_transform(df[["Yaş","Gelir"]]) #tablo içindeki yaş ve geliri standart hale getir saçmalakları ortadan kaldırır
###Lojistik işlerinde kullanılır  genelde

#df.drop("ID",axis=1,inplace=True)
#df["Gelir_Grubu"] = pd.cut(df["Gelir"],bins=[0,3000,5000,7000],labels=["Düşük","Orta","Yüksek"])
#df.to_excel("Kategorik_Gelir.xlsx",index=False)

#görselleştirme
plt.figure(figsize=(10,6))
plt.hist(df["Yaş"],bins=10,color="skyblue",edgecolor="black")
plt.title("yaş dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Frekans")
#plt.show()


plt.figure(figsize=(10,6))
sns.countplot(x="Gelir_Grubu",hue="Cinsiyet",data=df)

plt.title("Gelir grubu ve cinsiyet ilişkisi")
plt.xlabel("Gelir Grubu")
plt.ylabel("Cİnsiyet")
plt.show()