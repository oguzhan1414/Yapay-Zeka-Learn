import numpy as np
#####numpay içerde tek tip veri tipi tutar sadece int sadece float gibi
l1 = [1,2,3,4]
l2 = [2,4,5,6]

lst = []
for i in range(0,len(l1)):
    lst.append(l1[i]*l2[i])
print(lst)

arr1 = np.array([1,3,5,7])
arr2 = np.array([2,4,6,8])

print(arr1*arr2)

#Nump Arrayleri
lst = [1,2,3,4,5]
arr = np.array(lst)

np.zeros(5) # 5 elemandan oluşan 0 lardan oluşan numpy arreyi oluşturdu
np.ones(5) # 5 elamandan 1 lerden

np.arange(0,10,2) #0-10 arasında 2şer artacak şekilde
np.linspace(0,1,5) # 0-1 arasında 5 sayı
np.random.randint(0,100,5) # 0 -99 arası rastgele 5 sayı
np.random.normal(0,1,(2,3)) #2satırlık 3 sütünluk bir matris oluşturduk


#numpy attributes
arr1d = np.arange(0,10,2)
print(arr1d)
print(arr1d.shape)  #1 - tek boyutlu array
print(arr1d.ndim) #boyut bilgisi
print(arr1d.size) # toplam eleman sayısı


rastgeleArray=np.random.seed(42) #bu şekilde seed ile yarattığımız rastgele bir arrayi sürekli aynı şekilde görmemizi sağlar


#################Index seçimi

arr1d[:3] ##[0:3] arasını verir



####################################################
#NUMPY İLE MATEMATİKSEL İŞLEMLER
####################################################
arr = np.array([1,2,3,4,5])
arr*2
np.sum(arr)
np.mean(arr) #ortalmaa
np.median(arr)
np.std(arr) #standart sapmasını alır

######numpy ile koşullu işlemler
arr > 4
arr[arr > 4]
arr[arr != 4]


