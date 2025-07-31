#######################PYTHON TEMELLERİ#######################

print("Hello World")
print(33)
print(type(33.44))

# ATAMALAR VE DEĞİŞKENLER (Assignments & Variables)


a = 33
print(a)
b = "Hello AI"
c = 10
b = a * c
print(b)

#####Veri yapılarına giriş
# Integer (Tam Sayı)
x = 33
type(x)  # int olarak döner

# float
y = 33.34

# complex
z = 8j + 18
print(type(z))

# String (Karkater Dİzileri)
dizi = "Hello Python"

# Boolean (True / False)
isA = True

d = 22 < 1
print(d)  # böyle bir ifade false döner 22 sayısı 1 sayısından büyük olduğu için

# Listeler
l = [1, 2, 3, 4, "String", 3.2, False]  # Sıralıdır , Kapsayıcıdır , Değiştirilebilir

# Dictionary (Sözlükler)   Değiştirilebilir , Kapsayıdır , Sırasızdır  Key - Value Parametleri vardır keyden bir adet olabilir Name bir keydir jake ise bir value yani değerdir
dict = {"Name": "Jake",
        "Age": [27, 56],
        "Adress": "Downtown"}

# Tuple   Değiştirilemez , Kapsayıcıdır , Sıralıdır
t = ("Machine Learning", "Data Science")

# Set    Değiştirilebilir , Sırasızdır , Her Değerden bir adet olur (iki tane aynı değer girse bile 1 tanesi gözükür) , Kapsayıcıdır
s = {"Python", "Machine Learning", "Data Science", "Python"}
print(s)

#################################################################################

# Tipleri değiştirmek
a1 = 8  # Int
b1 = 11.2  # Float
z1 = 7j + 18  # Complex

float(a1)
int(b1)

# Karakter Dizilerinde Elemanlara Erişmek
name = "Python"
print(name[0])  # P yi ekrana yazdırırırz
print(name[-1])  # n yu ekrana yazdırırız
print(name[0:3])  # 0.indeksten 3.indekse kadar 3.indeks dahil değildir

# String içinde Karakter Sorgulaması   in
uzun_Dizi = "Bugün Python Öğrenmeye Başladım"
print("Python" in uzun_Dizi)

##########String Metodları
dir(str)  # string içindeki tüm metodları gösterir
text = "Hello AI Era"
text.upper()  # tüm karakterleri büyük harf yapar
text.lower()  # tüm karakteri küçük harf yapar
text.replace("l", "k")  # l stringini k ya çeviriri

text.upper().replace("L", "K")  # metodlar arka arkaya da kullanılabilir

text.split()  # stringi böler eğer bir veri vermezsek boşluklara böler ve çıktı olarak bir string döndürür
text.capitalize()  ## ilk harfi büyük yazar

len(text)  # stringin uzunluğunu dönderir int döner

##########Listeler(List) Metodları
lst = ["Data Science", 101, True, ["Kırmızı", "Sarı"]]

print(lst[1])

lst.append(33.44)  # listeye yeni eleman ekler
lst.pop(1)  # indekse göre eleman siler
#list.insert(0, "AI")  # indekse göre eleman ekler


############FONKSİYONLAR (KENDİ FONKSİYONUMUZU YAPALIM)

def multiply_three(number):
    print(number * 3)

multiply_three(5)

#Ön tanımlı argümanlar

def selamla(word ="Selam"):
    print(word,"Python")

selamla()
selamla("Merhaba")

#iki parametreli bir fonksiyon tanımlama

def cikarma(num1,num2):
    result = num1-num2
    print(result)

cikarma(20,5)
cikarma(num1=8,num2=3)


#Return : Fonksiyonun sonucunu kaydedip kullanmamızı sağlar (Fonksiyon çıktıları girdi olarak kullanmak için )
def toplama(num1,num2=5):
    sonuc = num1 + num2
    return sonuc

print(toplama(5))
toplam = toplama(12,7) #return ile fonksiyonun bir değişkene atamış olduk mantık bu kadar kolay aslında
print(toplam)


#Örnek : Öğrencilerin vize notlarının %40ını final notlarının %60ını ağırlık olarak alan bir fonksiyon yaz

def notSistemi(vize,final):
    result = (vize*0.4) + (final*0.6)
    return result

ogr1=notSistemi(40,70)
ogr2= notSistemi(80,20)
print(ogr1)
print(ogr2)


#######################################
#Koşullar
#######################################

#if condition
if 3 == 3: ###eğer bu ifade doğruysa alttaki ifade çalışır
    print("eşit")


#if-else condition
nick = "Oğuzhan"
if nick == "Oğuzhan":
    print("İsim Oğuzhan")
else:
    print("İsmi Oğuzhan Değil")

#if-elif-else condition
if nick == "Oğuzhan":
    print("İsim Oğuzhan")
elif nick == "Furkan":
    print("ismi FURKAN")
else:
    print("İsmi Oğuzhan Değil")


#Fonksiyonlarla koşul durumlarını birleştirelim
def check_name(name):
    if name == "Oğuzhan":
        print("isim Oğuzhan")
    else:
        print("isim oğuz değil")

check_name("Cabbar")


####################################
#Döngüler   bir yapı içinde tek tek gezmek istiyorsak döngü kullanılır
####################################
#for loop
dizi2 = ["Ali","Ayşe","Mehmet","Bartu"]

for isim in dizi2:
    print(isim)

for buyukHarf in dizi2:
    print(buyukHarf.upper())


#while loop
num = 5
while num<10: #burası true olduğu sürece döngü çalışır
    print(num)
    num = num+1


notes = [10,20,30,40,50,60,70,80]
index = 0
while index < len(notes):
    print(notes[index])
    index+=1