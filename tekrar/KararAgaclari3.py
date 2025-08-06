import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


data = {
    "size" : [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000, 5500, 6000,7500],
    "rooms" : [3, 3, 4, 4, 5, 5, 6, 6, 6, 7,8],
    "price" : [300, 320, 400, 450, 500, 540, 600, 620, 670, 700,820]
}

df = pd.DataFrame(data)

X= df[["size","rooms"]]
y= df["price"]

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeRegressor(random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

input_size = int(input("Metrekare girin"))
input_rooms = int(input("Oda sayısını girin"))
guess_price = model.predict([[input_size,input_rooms]])
print(f"Tahmini fiyat {guess_price[0]:.2f}")