import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    "Surface Roughness (μm)": surface_data,
    "Material Removal Rate (mm³/min)": mrr_data,
    "Kerf Angle (°)": kerf_data
}

# ------------------- Plotting -------------------
for name, data in datasets.items():
    df = pd.DataFrame(data)
    y_true = df['Exp.value'].values
    y_pred = df['Ensemble'].values

    # Compute 95% Prediction Interval
    residuals = y_true - y_pred
    std = np.std(residuals)
    pi_upper = y_pred + 1.96 * std
    pi_lower = y_pred - 1.96 * std

    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(y_true))

    plt.plot(x, y_true, "o-", color='blue', lw=2, markersize=5, label="Experimental")
    plt.plot(x, y_pred, "s--", color='red', lw=2, markersize=5, label="Ensemble Prediction")

    plt.fill_between(x, pi_lower, pi_upper, color='gray', alpha=0.3, label="95% Prediction Interval")

    plt.xlabel("Sample Index", fontsize=12, fontweight='bold')
    plt.ylabel(name, fontsize=12, fontweight='bold')
    plt.title(f"{name}: Experimental vs Ensemble Prediction", fontsize=14, fontweight='bold')

    # Metrics text
    textstr = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
