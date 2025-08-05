import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("matplot.xlsx") #verimizi excel den çekelim

##print(data.describe())
#print("*****************")
#print(data.info())
#print("*****************")


##########ÇİZGİ GRAFİĞİ
"""
plt.figure(figsize= (12,6))
#plt.plot(data["Satış"],data["Fiyat (TL)"]) genel yapıda bu şekildedir
plt.plot(data["Satış"],data["Fiyat (TL)"],color="green",linestyle="--",linewidth=2,marker="o")
plt.title("Ürün adı fiyat karşılaştırma")
plt.xlabel("Ürün adı")
plt.ylabel("Fiyat tl")
plt.grid(True)
"""
#########Çubuk grafiği
"""
plt.figure(figsize= (12,6))
plt.bar(data["Satış"],data["Fiyat (TL)"],color="skyblue")
plt.title("Satış Fiyat")
plt.xlabel("Satış ")
plt.ylabel("Fiyat")
plt.xticks(rotation=45) ##çok fazla verimiz olursa karışmaması adına x deki verilerimiz 45 derece döndürüyoruz

"""
#########Nokta Grafiği
"""
plt.figure(figsize= (12,6))
plt.scatter(data["Satış"],data["Fiyat (TL)"],color="skyblue")
plt.title("Satış Fiyat")
plt.xlabel("Satış ")
plt.ylabel("Fiyat")
plt.grid(True)
"""

#########Pasta Grafiği
"""
plt.figure(figsize= (12,6))
plt.pie(data["Satış"],labels=data["Kategori"],autopct="%1.1f%%", startangle=90)
plt.title("Satış Fiyat")
plt.axis("equal")
"""

#########Histogram Grafiği
"""
plt.figure(figsize= (12,6))
plt.hist(data["Kategori"],bins=20,color="green")
plt.title("Veri Dağılımı")

"""
#########Birden fazla grafiği aynı anda göstermek için:

plt.figure(figsize=(12,6))

# İlk alt grafiği (1 satır, 2 sütun, 1. grafik) oluştur
plt.subplot(1, 2, 1)

# Çubuk grafik: X ekseni 'Satış', Y ekseni 'Fiyat (TL)' olacak şekilde çiz
plt.bar(data["Satış"], data["Fiyat (TL)"], color="skyblue")
plt.title("Çubuk Grafiği")

# İkinci alt grafiği (1 satır, 2 sütun, 2. grafik) oluştur
plt.subplot(1,2,2)
plt.pie(data["Satış"],labels=data["Kategori"],autopct="%1.1f%%", startangle=90)
plt.title("Pasta Grafiği")

# Alt grafiklerin birbirine çakışmaması için aralıkları otomatik ayarla
plt.tight_layout()
plt.show()