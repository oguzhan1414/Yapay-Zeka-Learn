from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

model = Sequential()

# 1. Embedding: Kelimeleri sayısal vektörlere çevirir (NLP için şart)
# input_dim=1000 (Kelime dağarcığı), output_dim=64
model.add(Embedding(input_dim=1000, output_dim=64))

# 2. LSTM Katmanı
# 128 nöronluk hafıza hücresi.
model.add(LSTM(128, return_sequences=False)) # Tek bir çıktı versin (Sıradaki kelime ne?)

# 3. Çıktı Katmanı
model.add(Dense(1, activation='sigmoid')) # 0 veya 1 (Örn: Cümle olumlu mu olumsuz mu?)

model.summary()