import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Boş bir model oluştur
model = models.Sequential()

# 2. CNN Katmanları Ekleme
# Conv2D: Görüntü üzerinde filtre gezdirir (Özellik çıkarır)
# MaxPooling2D: Görüntüyü küçültür (Önemsiz detayları atar, işlem gücü kazandırır)

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))) # Giriş: 64x64 piksel renkli resim
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 3. Sınıflandırma Katmanı (Dense)
model.add(layers.Flatten()) # Görüntüyü düz bir vektöre çevir
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # 10 farklı sınıfı tahmin et (Kedi, köpek, kuş vb.)

# Modelin özetini gör
model.summary()