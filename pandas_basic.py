import numpy as np
import pandas as pd


###########Pandas series

"""
arr = np.array([1,2,3,4,5,6,7])
ser = pd.Series(arr)
print(ser)

print(ser.head()) ###serinin ilk 5 satırını gösterir
print(ser.tail()) ##serini son 5 satırını gösterir
"""


##############Pandas DateFrame

import seaborn as sns
df = sns.load_dataset("tips")
print(df.head())
print(df.columns) #kolanları gösteriri
print(df.dtypes) #her bir kolonun typenı dönderir
print(df.size) #boyutu
#print(df.values)
print(df.info) #veri hakkında satır ve sütün bilgilerini verir
print(df.describe())
print(df.isnull().sum()) #boş değer olup olmadığına bakmak için


#pandas fonksiyonları
df["total_bill"].head()  # "total_bill" (toplam hesap) sütunundaki ilk 5 veriyi gösterir

df["day"].value_counts()  # "day" sütunundaki her bir günün kaç kez tekrar ettiğini (frekansını) verir

df["day"].nunique()  # "day" sütunundaki kaç farklı (benzersiz) gün olduğunu söyler

df["day"].unique()  # "day" sütununda hangi günler geçtiğini (benzersiz değerleri) liste halinde verir


# Seçim ve Filtre İşlemleri (loc & iloc)
print(df[df["total_bill"]>10].head())  ###total bil 10 dan büyük olan ilk 5 veriyi getir


# loc
# df.loc[satır, sütun]
print(df.loc[0:3, "total_bill"]) #ilk 3 sütünu getir ve total_billi getir

# loc ile filtreleme
df.loc[df["total_bill"] > 10].head()
df.loc[df["total_bill"] > 10, "size"].head()
df.loc[df["total_bill"] > 10, ["smoker", "size"]].head() #10 dan büyük olan verileri ve sadece smoker ve size olan sütünları getir


# iloc
print(df.iloc[0]) #0 .satırı getirir
df.iloc[0, 2:4] #0.satırı getirir 2 ve 3 satırı getirir


###################################
# Gruplama işlemleri (Aggregation)
###################################
# groupby
df.groupby("day")["total_bill"].mean() ##günleri alıp günlerin fiyatların ortalaması alıp bakabiliyor

df.groupby("day").agg({
    "total_bill": "mean"
})

df.groupby("day").agg({ ###bu şekilde ile birden fazla satıra bakmamızı sağlar agg
    "total_bill": ["mean", "sum"],
    "tip": "mean"

})


# pivot table  groupby ile aynı işlemi yapar  gruplama yapar yani
df.pivot_table(index="day",values="total_bill" ,aggfunc="mean")
df.pivot_table(
    values=["total_bill", "tip"],
    index="day",
    aggfunc={
        "total_bill": ["mean", "sum"],
        "tip": "mean"
    }
)



####elimize bir veri geldiğinde ilk olarak bunları çalıştırmamız lazım verimizi iyi anlamzsak bir şey yapamayız bunu da böyle fonksiyona atıyabiliriz
##################################################
# KEŞİFÇİ VERİ ANALİZİ (EXPLORATORY DATA ANALYSIS)
##################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T) # sayısal değişkenlerin dağılım bilgisi


############################
# Kategorik Değişken Analizi
############################
cat_cols = ['sex', 'smoker', 'day', 'time']
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")



cat_summary(df, "day") # 1 değişkene bakmak için

# tüm kategorik sütunlarda döngü
for col in cat_cols:
    cat_summary(df, col)

##########################
# Numerik Değişken Analizi
##########################
num_cols = ['total_bill', 'tip', 'size']
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(dataframe=df, numerical_col="tip")  # 1 değişkene bakmak için

# tüm sayısal sütunlarda döngü
for col in num_cols:
    num_summary(df, col)

########################
# Hedef Değişken Analizi
########################
TARGET = "total_bill"
# Kategorik değişken ile hedef değişkenin ilişkisi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(dataframe=df,
                            target=TARGET,
                            categorical_col=col)