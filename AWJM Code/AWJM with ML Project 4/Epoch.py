import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== 1. Prepare data =====
surface_data = {
    'Exp.value': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.50,
                  3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                  4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.80],
    'Ensemble': [4.67, 3.27, 4.44, 4.95, 3.92, 4.99, 4.34, 4.63, 4.00, 3.61,
                 4.08, 4.53, 5.51, 3.66, 4.20, 4.74, 4.44, 4.32, 3.47, 4.26,
                 4.55, 5.29, 3.36, 4.44, 4.56, 5.20, 3.86]
}

mrr_data = {
    'Exp.value': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,
                  143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,
                  132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'Ensemble': [120.19, 126.98, 123.18, 112.97, 127.78, 98.15, 121.36, 110.81, 135.47, 120.34,
                 140.67, 124.83, 107.87, 145.89, 103.58, 113.40, 123.18, 119.27, 143.32, 128.00,
                 128.69, 100.57, 138.14, 123.18, 105.61, 105.06, 137.47]
}

kerf_data = {
    'Exp.value': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                  1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                  2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73],
    'Ensemble': [2.05, 1.52, 1.96, 2.20, 1.80, 2.23, 1.93, 2.08, 1.76, 1.66,
                 1.89, 1.99, 2.51, 1.68, 1.93, 2.11, 1.96, 1.94, 1.57, 1.90,
                 2.00, 2.37, 1.54, 1.96, 2.08, 2.32, 1.76]
}



datasets = {
    "Surface Roughness": surface_data,
    "Material Removal Rate": mrr_data,
    "Kerf Angle": kerf_data
}

# ===== 2. Dummy feature (sequence index) =====
num_samples = len(surface_data["Exp.value"])
X = np.arange(num_samples).reshape(-1, 1)

# ===== 3. Train NN and only plot training loss =====
def train_and_plot_loss(y_ensemble, target_name):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    history = model.fit(X, y_ensemble, epochs=500, batch_size=4, verbose=0, callbacks=[es])

    # Smooth training curve using moving average
    loss = np.array(history.history['loss'])
    smooth_loss = np.convolve(loss, np.ones(10)/10, mode='valid')

    # Plot smooth training loss (thinner line)
    plt.figure(figsize=(6, 3.8))
    plt.plot(smooth_loss, linewidth=1.5, color='blue')
    plt.xlabel("Epochs", fontsize=12, fontweight="bold")
    plt.ylabel("Loss (MSE)", fontsize=12, fontweight="bold")
    plt.title(f"{target_name} - Training Loss", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ===== 4. Run for all datasets =====
for name, data_dict in datasets.items():
    y_ensemble = np.array(data_dict["Ensemble"])
    train_and_plot_loss(y_ensemble, name)
