import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === MANUAL DATA ENTRY ===
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

# === Convert to DataFrames ===
surface = pd.DataFrame(surface_data)
mrr = pd.DataFrame(mrr_data)
kerf = pd.DataFrame(kerf_data)

# === Calculate Performance Metrics ===
performance_metrics = {}

for df, target_name in zip([surface, mrr, kerf],
                           ["Surface Roughness", "Material Removal Rate", "Kerf Angle"]):
    y_true, y_pred = df['Exp.value'], df['Ensemble']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    performance_metrics[target_name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# === Plot Function ===
def plot_actual_vs_pred(df, target_name, ax):
    y_true, y_pred = df['Exp.value'], df['Ensemble']
    metrics = performance_metrics[target_name]

    # Scatter plot
    ax.scatter(y_true, y_pred, color="royalblue", edgecolor="k", alpha=0.8, s=60)

    # Ideal fit line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label="Ideal Fit")

    # Labels
    ax.set_title(f"{target_name}\n(Ensemble Prediction)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Values", fontsize=10)
    ax.set_ylabel("Predicted Values", fontsize=10)

    # Metrics box
    metrics_text = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nR²: {metrics['R²']:.4f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

# === Plot All 3 Together ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_actual_vs_pred(surface, "Surface Roughness", axes[0])
plot_actual_vs_pred(mrr, "Material Removal Rate", axes[1])
plot_actual_vs_pred(kerf, "Kerf Angle", axes[2])

plt.suptitle("Actual vs Ensemble Predicted Values with Performance Metrics (RMSE, MAE, R²)",
             fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# === Print Summary Table ===
print("Ensemble Model Performance Summary")
print("=" * 50)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 50)
for target, metrics in performance_metrics.items():
    print(f"{target:<25} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R²']:<10.4f}")
print("=" * 50)
