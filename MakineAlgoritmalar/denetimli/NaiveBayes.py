import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. VERİ SETİ (Senaryo: Gelen Mesajlar)
# Spam ve Normal (Ham) mesajlar karışık
data = {
    'Mesaj': [
        "Tebrikler bedava tatil kazandınız hemen tıklayın",  # SPAM
        "Bugün dersten sonra buluşalım mı?",  # NORMAL
        "Fatura ödemeniz gecikmiştir lütfen arayın",  # SPAM (olabilir)
        "Yarınki toplantı saat 10:00'da unutma",  # NORMAL
        "Özel kampanya! %50 indirim şansı seni bekliyor",  # SPAM
        "Akşama eve gelirken ekmek alır mısın?",  # NORMAL
        "Acil nakit ihtiyacınız için hemen başvurun",  # SPAM
        "Proje dosyasını mail attım kontrol eder misin?",  # NORMAL
        "Sınırlı süre için büyük fırsat kaçırma",  # SPAM
        "Hafta sonu sinemaya gidelim mi?"  # NORMAL
    ],
    'Etiket': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: SPAM, 0: NORMAL
}

df = pd.DataFrame(data)

# 2. VERİYİ SAYISALLAŞTIRMA (Bag of Words)
# Bilgisayar "tatil" kelimesini anlamaz, "tatil"in geçtiği sıklığı anlar.
# CountVectorizer, kelimeleri sayar ve bir matrise dönüştürür.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Mesaj'])  # Mesajları sayısal vektörlere çevir
y = df['Etiket']

# X'in neye benzediğini anlamak için (Kelime Havuzu)
# print(vectorizer.get_feature_names_out()) # Kelime listesini görmek istersen açabilirsin

# 3. EĞİTİM VE TEST AYRIMI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL KURULUMU
# Metin verileri için genelde 'MultinomialNB' kullanılır.
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. TAHMİN VE DEĞERLENDİRME
y_pred = model.predict(X_test)

print(f"Model Doğruluğu: %{accuracy_score(y_test, y_pred) * 100:.2f}")

# 6. GÖRSELLEŞTİRME (Spam Olasılıkları)
# Modelin hangi kelimeleri 'Spam' olarak işaretlediğini anlamak için
# Basit bir Confusion Matrix çizelim
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
plt.title('Spam Filtresi Başarısı')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')
plt.show()

# 7. İNTERAKTİF SPAM KONTROL ROBOTU
print("\n--- SPAM KONTROL SİSTEMİ ---")
print("Bir mesaj yazın, spam olup olmadığını söyleyeyim.")

while True:
    user_input = input("\nMesajınız (Çıkış için 'q'): ")
    if user_input.lower() == 'q':
        break

    # DİKKAT: Kullanıcının girdiği metni de modelin anladığı dile (vektöre) çevirmeliyiz!
    # Burada 'transform' kullanıyoruz, 'fit' değil. Çünkü model kelimeleri zaten öğrendi.
    input_vector = vectorizer.transform([user_input])

    tahmin = model.predict(input_vector)[0]
    olasilik = model.predict_proba(input_vector)[0]  # [Normal Olasılığı, Spam Olasılığı]

    if tahmin == 1:
        print(f"🚫 UYARI: Bu mesaj **SPAM** olabilir! (Spam İhtimali: %{olasilik[1] * 100:.1f})")
    else:
        print(f"✅ GÜVENLİ: Bu normal bir mesaj. (Güvenli İhtimali: %{olasilik[0] * 100:.1f})")