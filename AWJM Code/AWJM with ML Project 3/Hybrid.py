import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- AWJM Data -------------------
df = pd.DataFrame({
    'surface_roughness_um': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                             3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                             4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'material_removal_rate_mm3_min': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63,
                                      121.6, 109.81, 137.08, 117.28, 143.3, 125.7,
                                      106.95, 148.41, 102.02, 115.2, 122.4, 119.69,
                                      145.92, 131.74, 132.71, 99.7, 140.98, 122.4,
                                      104.26, 103.66, 139.85],
    'kerf_angle_deg': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                       1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                       2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
})

# ------------------- Ensemble Predictions -------------------
y_pred = pd.DataFrame({
    'surface_roughness_um': [4.65, 3.22, 4.45, 4.98, 3.90, 5.00, 4.32, 4.60, 4.01, 3.62,
                 4.05, 4.56, 5.50, 3.67, 4.19, 4.76, 4.45, 4.35, 3.48, 4.23,
                 4.57, 5.29, 3.38, 4.45, 4.55, 5.23, 3.86],
    'material_removal_rate_mm3_min': [119.91, 126.48, 123.15, 113.14, 127.55, 97.70, 121.78, 110.90, 136.23, 119.61,
                 141.41, 125.05, 107.78, 146.81, 103.09, 114.79, 123.15, 119.10, 144.08, 128.20,
                 129.19, 100.24, 139.20, 123.15, 105.50, 104.67, 137.92],
    'kerf_angle_deg': [2.05, 1.52, 1.97, 2.22, 1.78, 2.23, 1.91, 2.06, 1.77, 1.67,
                 1.87, 2.02, 2.49, 1.68, 1.92, 2.13, 1.97, 1.96, 1.59, 1.90,
                 2.00, 2.37, 1.54, 1.97, 2.08, 2.32, 1.76]
})

# ------------------- Metrics -------------------
metrics = {
    'surface_roughness_um': {'RMSE': 0.0706, 'MAE': 0.0593, 'R2': 0.9872},
    'material_removal_rate_mm3_min': {'RMSE': 1.4826, 'MAE': 1.1811, 'R2': 0.9894},
    'kerf_angle_deg': {'RMSE': 0.0437, 'MAE': 0.0363, 'R2': 0.9751}
}


# ------------------- Function to Plot Predictions -------------------
def plot_prediction(y_true, y_hat, target_name, units, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_hat, alpha=0.7, color=color, edgecolors='k', s=60)

    # Perfect prediction line
    min_val, max_val = min(y_true) * 0.95, max(y_true) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

    # 95% Prediction Interval
    residuals = y_true - y_hat
    std = np.std(residuals)
    pi_upper = y_hat + 1.96 * std
    pi_lower = y_hat - 1.96 * std
    sort_idx = np.argsort(y_true)
    ax.fill_between(np.array(y_true)[sort_idx], pi_lower[sort_idx], pi_upper[sort_idx],
                    color='gray', alpha=0.3, label='95% Prediction Interval')

    # Labels & Title
    ax.set_xlabel(f'Measured {target_name} ({units})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {target_name} ({units})', fontsize=12, fontweight='bold')
    ax.set_title(f'{target_name} Prediction', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)

    # Metrics text
    txt = metrics[target_name]
    textstr = f"R² = {txt['R2']:.4f}\nRMSE = {txt['RMSE']:.4f}\nMAE = {txt['MAE']:.4f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ------------------- Plot All -------------------
plot_prediction(df['surface_roughness_um'], y_pred['surface_roughness_um'],
                'surface_roughness_um', 'μm', 'blue')
plot_prediction(df['material_removal_rate_mm3_min'], y_pred['material_removal_rate_mm3_min'],
                'material_removal_rate_mm3_min', 'mm³/min', 'green')
plot_prediction(df['kerf_angle_deg'], y_pred['kerf_angle_deg'],
                'kerf_angle_deg', '°', 'purple')
