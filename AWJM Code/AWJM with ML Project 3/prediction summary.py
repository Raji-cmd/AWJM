import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === MANUAL DATA ENTRY ===
# Surface Roughness Data
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
