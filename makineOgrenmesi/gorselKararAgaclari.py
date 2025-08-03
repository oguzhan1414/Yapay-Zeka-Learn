import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("karar_agaci_veri_100.xlsx")

#yaş ile hastalık arasındaki ilişkiyi görselleştirme


plt.figure(figsize=(10,6))
sns.histplot(data=df,x="Yas",hue="Hastalik",multiple="stack",kde=False)
plt.title("Yaş Dağılımı ve Hastalık Durumu")
plt.xlabel("Yaş")
plt.ylabel("Kişi Sayısı")
#plt.show()


plt.figure(figsize=(10,6))
sns.histplot(data=df,x="Kan_Basinci",hue="Hastalik",multiple="stack",kde=False)
plt.title("Kan Basıncı ve Hastalık Durumu")
plt.xlabel("Kan Basıncı")
plt.ylabel("Kişi Sayısı")
#plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=df,x="Kolesterol",hue="Hastalik",multiple="stack",kde=False)
plt.title("Kolesterol ve Hastalık Durumu")
plt.xlabel("Kolesterol")
plt.ylabel("Kişi Sayısı")
plt.show()