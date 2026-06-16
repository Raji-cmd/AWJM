import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Data -------------------
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
