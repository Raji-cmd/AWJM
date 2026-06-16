import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== 1. Prepare data =====
data = {
    'Exp_value':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.5,3.97,4.63,5.56,3.62,4.08,4.75,4.46,4.41,3.42,4.27,
                  4.67,5.36,3.29,4.46,4.53,5.23,3.8],
    'Ridge':[4.49,3.37,4.34,5.03,3.98,4.96,4.26,4.7,3.85,3.65,4.33,4.42,5.64,3.72,4.35,4.67,4.34,4.25,3.38,4.19,
             4.43,5.3,3.37,4.34,4.69,5.31,3.99],
    'RandomForest':[4.82,3.43,4.47,4.9,3.78,5.05,4.37,4.63,3.87,3.6,3.93,4.56,5.47,3.65,4.23,4.67,4.47,4.41,3.52,4.34,
                     4.63,5.34,3.45,4.47,4.56,5.21,3.78],
    'XGBoost':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.5,3.97,4.63,5.56,3.62,4.08,4.75,4.46,4.41,3.42,4.27,
                4.67,5.36,3.29,4.46,4.53,5.23,3.8],
    'NeuralNet':[4.49,3.14,4.39,5.08,3.85,5.04,4.32,4.62,3.86,3.6,4.1,4.44,5.59,3.69,4.14,4.69,4.39,4.3,3.45,4.26,
                 4.41,5.45,3.35,4.39,4.59,5.3,3.91]
}

df = pd.DataFrame(data)

# ===== 2. Compute ensemble =====
model_cols = ['Ridge', 'RandomForest', 'XGBoost', 'NeuralNet']
weights = np.array([0.15, 0.25, 0.30, 0.30])
df['Ensemble'] = df[model_cols].values @ weights


print(df[['Exp_value', 'Ensemble']])

# ===== 3. Prepare features and target =====
# Example: Using index as feature (replace with actual inputs like Pulse, Current, Voltage)
X = np.arange(len(df)).reshape(-1,1)  # dummy features
y = df['Ensemble'].values

# ===== 4. Define Neural Network =====
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# ===== 5. Early stopping to find optimal epoch =====
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

history = model.fit(X, y, epochs=500, batch_size=4, callbacks=[early_stop], verbose=1)

# ===== 6. Plot training loss =====
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error')

# ✅ X-axis ticks (20, 40, 60…)
plt.xticks(np.arange(0, len(history.history['loss'])+1, 20))

# ✅ Y-axis ticks (0.0, 2.5, 5.0, 7.5…)
plt.yticks(np.arange(0, max(history.history['loss'])+2.5, 2.5))
# ✅ Axis labels
plt.xlabel("Epochs", fontsize=15, fontweight='bold')
plt.ylabel("Mean Square Error", fontsize=15, fontweight='bold')

# ✅ Tick label sizes
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# ✅ Title
plt.title("Training And Validation Loss Ensemble Surface Roughness", fontsize=15, fontweight='bold')
plt.show()

# ===== 7. Save results to Excel =====
df['NN_Prediction'] = model.predict(X)
df.to_excel("Surface_Roughness_Ensemble.xlsx", index=False)
print("✅ Saved: Surface_Roughness_Ensemble.xlsx")
