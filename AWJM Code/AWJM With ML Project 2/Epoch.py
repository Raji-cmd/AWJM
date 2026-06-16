import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== 1. Prepare data =====
surface_data = {
    'Exp.value': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                  3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                  4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'Ensemble': [4.65, 3.40, 4.42, 4.91, 3.93, 4.96, 4.32, 4.63, 3.95, 3.67,
                 4.12, 4.51, 5.45, 3.69, 4.27, 4.71, 4.42, 4.32, 3.49, 4.24,
                 4.52, 5.24, 3.43, 4.42, 4.57, 5.18, 3.91]
}

mrr_data = {
    'Exp.value': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,
                  143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,
                  132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'Ensemble': [120.37, 127.35, 122.82, 114.01, 128.66, 98.24, 121.47, 111.54, 134.56, 122.61,
                 139.15, 124.18, 108.54, 144.58, 104.32, 112.91, 122.82, 119.77, 141.91, 126.16,
                 126.52, 100.99, 136.36, 122.82, 106.39, 105.97, 136.02]
}

kerf_data = {
    'Exp.value': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                  1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                  2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73],
    'Ensemble': [2.06, 1.59, 1.96, 2.17, 1.81, 2.21, 1.92, 2.07, 1.75, 1.69,
                 1.89, 2.00, 2.43, 1.68, 1.96, 2.10, 1.96, 1.93, 1.59, 1.90,
                 2.00, 2.34, 1.59, 1.96, 2.07, 2.29, 1.78]

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
