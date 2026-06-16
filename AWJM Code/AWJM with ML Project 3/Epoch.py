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
    'Ensemble': [4.65, 3.22, 4.45, 4.98, 3.90, 5.00, 4.32, 4.60, 4.01, 3.62,
                 4.05, 4.56, 5.50, 3.67, 4.19, 4.76, 4.45, 4.35, 3.48, 4.23,
                 4.57, 5.29, 3.38, 4.45, 4.55, 5.23, 3.86]
}

mrr_data = {
    'Exp.value': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,
                  143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,
                  132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'Ensemble': [119.91, 126.48, 123.15, 113.14, 127.55, 97.70, 121.78, 110.90, 136.23, 119.61,
                 141.41, 125.05, 107.78, 146.81, 103.09, 114.79, 123.15, 119.10, 144.08, 128.20,
                 129.19, 100.24, 139.20, 123.15, 105.50, 104.67, 137.92]
}

kerf_data = {
    'Exp.value': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                  1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                  2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73],
    'Ensemble': [2.05, 1.52, 1.97, 2.22, 1.78, 2.23, 1.91, 2.06, 1.77, 1.67,
                 1.87, 2.02, 2.49, 1.68, 1.92, 2.13, 1.97, 1.96, 1.59, 1.90,
                 2.00, 2.37, 1.54, 1.97, 2.08, 2.32, 1.76]

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
