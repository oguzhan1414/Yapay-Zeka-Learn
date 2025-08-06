from sklearn.preprocessing import MinMaxScaler
import numpy as np


"""Min-Max Scaling
# Original data
data = np.array([[50, 30], [60, 90], [70, 100]])
# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Scale the data
scaled_data = scaler.fit_transform(data)
print("Scaled Data using Min-Max Scaling:\n", scaled_data)
"""


from sklearn.preprocessing import StandardScaler
""" Standardization
# Initialize StandardScaler
scaler = StandardScaler()
# Scale the data
scaled_data = scaler.fit_transform(data)
print("Scaled Data using Standardization:\n", scaled_data)
"""
