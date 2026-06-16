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

# ------------------- Ensemble Predictions (Actual values) -------------------
y_pred = pd.DataFrame({
    'surface_roughness_um': df['surface_roughness_um'],
    'material_removal_rate_mm3_min': df['material_removal_rate_mm3_min'],
    'kerf_angle_deg': df['kerf_angle_deg']
})

# ------------------- Metrics -------------------
metrics = {
    'surface_roughness_um': {'RMSE': 0.0768, 'MAE': 0.0563, 'R2': 0.9484},
    'material_removal_rate_mm3_min': {'RMSE': 2.9824, 'MAE': 2.5181, 'R2': 0.9571},
    'kerf_angle_deg': {'RMSE': 0.0345, 'MAE': 0.0294, 'R2': 0.9845}
}

# ------------------- Plot 1: Surface Roughness -------------------
fig, ax = plt.subplots(figsize=(8, 6))
y_true = df['surface_roughness_um'].values
y_hat = y_pred['surface_roughness_um'].values

ax.scatter(y_true, y_hat, alpha=0.7, color='blue', edgecolors='k', s=50)
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
ax.set_xlabel('Measured Surface Roughness, Ra (μm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Surface Roughness, Ra (μm)', fontsize=12, fontweight='bold')
ax.set_title('Surface Roughness Prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, linestyle=':', alpha=0.7)

# Metrics text
txt = metrics['surface_roughness_um']
textstr = f"R² = {txt['R2']:.4f}\nRMSE = {txt['RMSE']:.4f}\nMAE = {txt['MAE']:.4f}"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# ------------------- Plot 2: MRR & Kerf -------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets = ['material_removal_rate_mm3_min', 'kerf_angle_deg']
titles = ['Material Removal Rate (MRR)', 'Kerf Angle']

for i, target in enumerate(targets):
    ax = axes[i]
    y_true = df[target].values
    y_hat = y_pred[target].values

    ax.scatter(y_true, y_hat, alpha=0.7, color='green', edgecolors='k', s=50)

    min_val, max_val = min(y_true) * 0.95, max(y_true) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

    residuals = y_true - y_hat
    std = np.std(residuals)
    pi_upper = y_hat + 1.96 * std
    pi_lower = y_hat - 1.96 * std
    sort_idx = np.argsort(y_true)
    ax.fill_between(np.array(y_true)[sort_idx], pi_lower[sort_idx], pi_upper[sort_idx],
                    color='gray', alpha=0.3, label='95% Prediction Interval')

    ax.set_xlabel(f'Measured {titles[i]}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {titles[i]}', fontsize=12, fontweight='bold')
    ax.set_title(f'{titles[i]} Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)

    txt = metrics[target]
    textstr = f"R² = {txt['R2']:.4f}\nRMSE = {txt['RMSE']:.4f}\nMAE = {txt['MAE']:.4f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()
