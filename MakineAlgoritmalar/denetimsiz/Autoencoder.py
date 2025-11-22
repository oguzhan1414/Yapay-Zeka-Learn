from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Giriş Boyutu (Örneğin 784 piksellik bir resim vektörü)
input_img = Input(shape=(784,))

# 1. Encoder (Sıkıştırma)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(32, activation='relu')(encoded) # Veriyi 32 sayıya kadar sıkıştırdık!

# 2. Decoder (Geri Oluşturma)
decoded = Dense(128, activation='relu')(encoded)
output_img = Dense(784, activation='sigmoid')(decoded) # Tekrar 784 piksele çıkardık

# Modeli Birleştir
autoencoder = Model(input_img, output_img)

# Modeli eğitirken: GİRDİ = Gürültülü Resim, HEDEF = Temiz Resim
# autoencoder.fit(x_train_noisy, x_train_clean, ...)