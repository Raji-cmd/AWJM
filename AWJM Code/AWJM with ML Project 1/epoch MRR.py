import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== 1. Prepare data =====
data = {
    'Exp_value': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95, 148.41,
                  102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'Ridge Regression': [120.2, 126.7, 122.19, 117.72, 130.56, 94.96, 121.22, 113.82, 134.08, 126.66, 141.46, 123.17, 109.76,
                         149.42, 102.93, 114.27, 122.19, 119.47, 145.52, 124.18, 124.92, 98.87, 138.53, 122.19, 106.83, 105.86, 137.55],
    'RandomForest': [120.75, 128.46, 122.44, 110.36, 127.92, 98.86, 121.92, 109.18, 132.23, 121.02, 142.51, 124.45, 106.46,
                     147.08, 102.52, 109.07, 122.44, 120.83, 144.94, 128.84, 129.08, 100.41, 137.83, 122.44, 103.9, 104.98, 139.56],
    'XGBoost': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95,
                148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'NeuralNet': [111.89, 130.38, 114.6, 120.04, 130.47, 99.76, 113.21, 114.01, 129.74, 127.25, 148.3, 116, 111.62, 154.31,
                  107.29, 104.51, 114.6, 111.44, 152.27, 117.32, 117.76, 104.47, 146.16, 114.6, 109.41, 110.45, 142.91]
}

df = pd.DataFrame(data)

# ===== 2. Compute ensemble =====
model_cols = ['Ridge Regression', 'RandomForest', 'XGBoost', 'NeuralNet']
weights = np.array([0.15, 0.25, 0.30, 0.30])
df['Ensemble'] = df[model_cols].values @ weights

print(df[['Exp_value', 'Ensemble']])

# ===== 3. Prepare features and target =====
# Example: using index as feature (replace with actual input parameters like Pulse, Current, Voltage)
X = np.arange(len(df)).reshape(-1,1)  # dummy features
y = df['Ensemble'].values

# ===== 4. Define Neural Network =====
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# ===== 5. Early stopping to find optimal epochs =====
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

history = model.fit(X, y, epochs=500, batch_size=4, callbacks=[early_stop], verbose=1)

# ===== 6. Plot training loss =====
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error')
# ✅ Axis labels
plt.xlabel("Epochs", fontsize=15, fontweight='bold')
plt.ylabel("Mean Square Error", fontsize=15, fontweight='bold')
plt.title("Training And Validation Loss Ensemble MRR", fontsize=15, fontweight='bold')

plt.show()

