import numpy as np
import pandas as pd

df = pd.DataFrame(data=np.random.rand(5, 5), index=["A", "B", "C", "D", "E"],
                  columns=['Columns1', 'Columns2', 'Columns3', 'Columns4', 'Columns5'])
#print(df)
df["Colums6"] = pd.Series(np.random.rand(5), index=["A", "B", "C", "D", "E"])
##print(df)
df["Colums7"] = df["Columns3"] + df["Columns4"]
#3print(df)

df.drop("Colums7", axis=1, inplace=True)  #Axis : 0 satırları 1 sütünları temsil eder   inplace : işlemin kalıcı olup olmadığını döner true bu işlemi kalıcı olarka siler

#print(df)
print("************************************")

filter_df = df[df["Columns2"]>0.5]
#print(df[df.index.isin(["B","D"])])


arr = np.array([[10,20,np.nan],[3,np.nan,np.nan],[13,np.nan,np.nan]])
df2 = pd.DataFrame(data=arr,index=["index1","index2","index3"],columns=["Column1","Colum2","Colum3"])
#print(df2.isnull()) #Hangi Hücreler Eksik?
##print(df2.isnull().sum())  #Kaç Tane Eksik Veri Var?
#print(df2)
#df.dropna()   df.dropna(axis = 1)    df.dropna(thresh = 2)   df.fillna(value = 0)

#print(df2.dropna(thresh=2))
#print(df2)
#print(df2.fillna("Bilinmiyor"))
#print("************************************")

data = {'Job': ['Data Mining','CEO','Lawyer','Lawyer','Data Mining','CEO'],
        'Labouring': ['Immanuel','Jeff','Olivia','Maria','Walker','Obi-Wan'],
        'Salary': [4500,30000,6000,5250,5000,35000]
        }
df3 = pd.DataFrame(data)
salaryGroup = df3.groupby('Salary') ###maaşa göre gruplandırdık
print(salaryGroup.sum())
print(salaryGroup.min())
print(df3.groupby('Job'))    ###işlere göre sıralama yapma
print(df3.groupby('Job').sum().loc['CEO']) #‘CEO’ların maaşların toplamı
print(df3.groupby("Job").min()) #işlere göre sıralar en düşük maaş hangisindeyse onu getirir

print(df3.head())
print(df3.describe())
print(df3.info())
print(df3["Job"].unique())
print(df3["Job"].value_counts())
print(df3["Job"])