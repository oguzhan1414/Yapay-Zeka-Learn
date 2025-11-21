import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. VERİ SETİ (Senaryo: İşe Alım Kararı)
# Özellikler: [Tecrübe Yılı, Eğitim Seviyesi (0:Lisans, 1:Yüksek, 2:Doktora), Mülakat Puanı]
data = {
    'Tecrube': [1,   10,  2,   5,   4,   8,   0,   3,   6,   7,   2,   9],
    'Egitim':  [0,   2,   0,   1,   0,   2,   0,   1,   1,   2,   0,   1], # 0:Lisans, 1:YL, 2:Dr
    'Puan':    [50,  95,  45,  80,  60,  90,  40,  70,  85,  88,  55,  92],
    'Ise_Alindi': [0,   1,   0,   1,   0,   1,   0,   0,   1,   1,   0,   1] # 0:Hayır, 1:Evet
}

df = pd.DataFrame(data)

X = df[['Tecrube', 'Egitim', 'Puan']]
y = df['Ise_Alindi']

# 2. EĞİTİM VE TEST AYRIMI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL KURULUMU
# criterion='gini': Ayrımı neye göre yapacağını belirler (standarttır).
# max_depth=3: karmaşıklaşm Ağacın çokasını ve ezberlemesini (overfitting) önlemek için derinliği sınırlarız.
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 4. TAHMİN VE SKOR
y_pred = tree_model.predict(X_test)
print(f"Model Doğruluğu: %{accuracy_score(y_test, y_pred) * 100:.2f}")

# 5. AĞACI GÖRSELLEŞTİRME (En Güzel Kısmı)




plt.figure(figsize=(12, 8))
plot_tree(tree_model, 
          feature_names=['Tecrube', 'Egitim', 'Puan'],  
          class_names=['Red', 'Kabul'],
          filled=True, 
          rounded=True)
plt.title("İşe Alım Karar Ağacı")
plt.show()

# 6. YENİ ADAY TAHMİNİ
print("\n--- İK Karar Robotu ---")
# Örnek Aday: 4 yıl tecrübeli, Yüksek Lisans (1) yapmış, mülakattan 78 almış.
yeni_aday = [[4, 1, 78]] 

sonuc = tree_model.predict(yeni_aday)

print(f"Aday Özellikleri: {yeni_aday}")
if sonuc[0] == 1:
    print("Karar: ✅ İŞE ALINDI")
else:
    print("Karar: ❌ REDDEDİLDİ")