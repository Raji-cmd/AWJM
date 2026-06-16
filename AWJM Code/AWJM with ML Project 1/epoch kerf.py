import pandas as pd
import numpy as np

# ===== 1. Prepare data =====
data = {
    'Exp_value': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                  1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                  2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73],
    'Ridge Regression': [2.02, 1.5, 1.95, 2.27, 1.8, 2.23, 1.91, 2.1, 1.73, 1.64,
              1.96, 2.0, 2.55, 1.68, 1.95, 2.1, 1.95, 1.91, 1.51, 1.88,
              1.99, 2.4, 1.52, 1.95, 2.12, 2.38, 1.79],
    'RandomForest': [2.12, 1.51, 1.98, 2.21, 1.74, 2.26, 1.95, 2.11, 1.75, 1.65,
                     1.83, 2.02, 2.51, 1.6, 1.96, 2.11, 1.98, 1.96, 1.54, 1.95,
                     2.03, 2.41, 1.51, 1.98, 2.08, 2.34, 1.71],
    'XGBoost': [2.079999924, 1.440000057, 1.980000019, 2.25, 1.75, 2.25,
                1.950000048, 2.089999914, 1.830000043, 1.590000033, 1.860000014,
                2.039999962, 2.569999933, 1.639999986, 1.860000014, 2.150000095,
                1.980000019, 1.960000038, 1.519999981, 1.929999948, 2.049999952,
                2.430000067, 1.480000019, 1.980000019, 2.079999924, 2.339999914,
                1.730000019],
    'NeuralNet': [2.01, 1.51, 1.88, 2.27, 1.79, 2.31, 1.88, 2.15, 1.7, 1.6,
                  1.91, 1.9, 2.6, 1.72, 1.86, 2.03, 1.88, 1.86, 1.57, 1.75,
                  1.88, 2.46, 1.52, 1.88, 2.15, 2.39, 1.78]
}

df = pd.DataFrame(data)

# ===== 2. Compute ensemble =====
model_cols = ['Ridge Regression', 'RandomForest', 'XGBoost', 'NeuralNet']
weights = np.array([0.15, 0.25, 0.30, 0.30])
df['Ensemble'] = df[model_cols].values @ weights

print(df[['Exp_value', 'Ensemble']])

# ===== 3. Prepare features and target =====
# Example: using index as feature (replace with actual input parameters if available)
X = np.arange(len(df)).reshape(-1,1)
y = df['Ensemble'].values

# ===== 4. Define Neural Network =====
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# ✅ Axis labels
plt.xlabel("Epochs", fontsize=15, fontweight='bold')
plt.ylabel("Mean Square Error", fontsize=15, fontweight='bold')

# ✅ Tick label sizes
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# ✅ Title
plt.title("Training And Validation Loss Ensemble Kerf Angle", fontsize=15, fontweight='bold')

plt.show()

