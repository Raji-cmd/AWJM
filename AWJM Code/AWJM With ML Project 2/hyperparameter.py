import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === MANUAL DATA ENTRY ===
# Surface Roughness Data
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

# Create DataFrames
surface = pd.DataFrame(surface_data)
mrr = pd.DataFrame(mrr_data)
kerf = pd.DataFrame(kerf_data)

# === Calculate Performance Metrics (RMSE instead of MSE) ===
performance_metrics = {}

for df, target_name in zip([surface, mrr, kerf],
                           ["Surface Roughness", "Material Removal Rate", "Kerf Angle"]):
    y_true = df['Exp.value']
    y_pred = df['Ensemble']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    performance_metrics[target_name] = {"RMSE": rmse, "MAE": mae, "R²": r2}


# === Enhanced Plot Function ===
def plot_actual_vs_pred(df, target_name, ax):
    actual_col = "Exp.value"
    predicted_col = "Ensemble"
    metrics = performance_metrics[target_name]

    # Scatter plot
    ax.scatter(df[actual_col], df[predicted_col], color="blue",
               edgecolor="k", alpha=0.7, s=60)

    # Perfect prediction line
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="red", linestyle="--", linewidth=1.5, label="Ideal Fit")

    # Labels and title
    ax.set_title(f"{target_name}\nEnsemble Prediction", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    # Metrics text box
    metrics_text = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nR²: {metrics['R²']:.4f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)


# === Create Figure with 3 Subplots ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_actual_vs_pred(surface, "Surface Roughness", axes[0])
plot_actual_vs_pred(mrr, "Material Removal Rate", axes[1])
plot_actual_vs_pred(kerf, "Kerf Angle", axes[2])

plt.suptitle("Actual vs Ensemble Predicted Values with Performance Metrics (RMSE, MAE, R²)",
             fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# === Print Performance Summary Table ===
print("Ensemble Model Performance Summary")
print("=" * 50)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 50)
for target, metrics in performance_metrics.items():
    print(f"{target:<25} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R²']:<10.4f}")
print("=" * 50)
