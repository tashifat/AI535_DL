# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:03:13 2024

@author: NILA_HOME
"""

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Attention, Concatenate, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt

# Function to create dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Load your dataset (replace with your actual data loading code)
data = pd.read_csv('refined_dataset/B33_discharge_soh.csv')
dataset = data['SOH'].values.reshape(-1, 1)

# Normalize the dataset
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)

# Split into train and test datasets
train_size = int(len(dataset_scaled) * 0.9)
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:]

# Create datasets
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))
testX = np.reshape(testX, (testX.shape[0], look_back, 1))

# Build Attention LSTM model
input_layer = Input(shape=(look_back, 1))
lstm_layer = LSTM(64, return_sequences=True)(input_layer)
attention = Attention()([lstm_layer, lstm_layer])
context_vector = Concatenate()([lstm_layer, attention])
dropout_layer = Dropout(0.2)(context_vector)
dense_layer = Dense(64, activation='relu')(dropout_layer)
output_layer = Dense(1)(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mae')

# Train the model
history = model.fit(
    trainX, trainY,
    epochs=40,
    batch_size=64,
    validation_data=(testX, testY),
    verbose=1,
    shuffle=False
)
#%%
# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions
yhat = model.predict(testX)

yhat = yhat[:, -1, :]  # Select the last time step for each sequence


yhat_inverse = scaler.inverse_transform(yhat)

testY = testY.reshape(-1, 1)
testY_inverse = scaler.inverse_transform(testY)

# Check for matching shapes
if yhat_inverse.shape != testY_inverse.shape:
    yhat_inverse = yhat_inverse[:testY_inverse.shape[0], :]  # Align shapes if needed

rmse = math.sqrt(mean_squared_error(testY_inverse, yhat_inverse))
mae = mean_absolute_error(testY_inverse, yhat_inverse)

print(f'Test RMSE: {rmse:.3f}')
print(f'Test MAE: {mae:.3f}')


# Visualization
plt.plot(testY_inverse, label='Real SOH')
plt.plot(yhat_inverse, label='Predicted SOH')
plt.legend()
plt.show()
