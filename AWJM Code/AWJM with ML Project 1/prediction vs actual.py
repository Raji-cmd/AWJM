import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === MANUAL DATA ENTRY ===
# Surface Roughness Data
surface_data = {
    'Exp.value': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46,
                  4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'Ensemble': [4.62, 3.36, 4.44, 4.95, 3.94, 4.97, 4.37, 4.62, 4.0, 3.71, 4.12, 4.53, 5.45, 3.8, 4.16, 4.69, 4.44,
                 4.38, 3.56, 4.3, 4.53, 5.27, 3.48, 4.44, 4.57, 5.18, 3.94]
}

# Material Removal Rate Data
mrr_data = {
    'Exp.value': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95,
                  148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66,
                  139.85],
    'Ensemble': [119.34, 126.36, 121.65, 115.80, 127.92, 100.59, 120.42, 113.10, 132.54, 122.67, 141.76, 122.78, 110.27,
                 147.86, 106.01, 112.92, 121.65, 118.70, 144.76, 125.65, 126.72, 103.08, 139.27, 121.65, 107.99, 107.85,
                 138.22]
}

# Kerf Angle Data
kerf_data = {
    'Exp_value': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59, 1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98,
                  1.96, 1.52, 1.93, 2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73],
    'Ensemble': [2.05, 1.54, 1.96, 2.22, 1.79, 2.23, 1.94, 2.09, 1.78, 1.68, 1.90, 2.00, 2.50, 1.70, 1.90, 2.10, 1.96,
                 1.94, 1.59, 1.90, 1.99, 2.37, 1.56, 1.96, 2.09, 2.32, 1.78]
}

# Performance Metrics (RMSE instead of MSE)
performance_metrics = {
    "Surface Roughness": {"RMSE": 0.0768, "MAE": 0.0563, "R²": 0.9848},
    "Material Removal Rate": {"RMSE": 2.9824, "MAE": 2.5181, "R²": 0.9571},
    "Kerf Angle": {"RMSE": 0.0345, "MAE": 0.0294, "R²": 0.9845}
}

# Create DataFrames
surface = pd.DataFrame(surface_data)
mrr = pd.DataFrame(mrr_data)
kerf = pd.DataFrame(kerf_data)

# === Enhanced Plot Function ===
def plot_actual_vs_pred(df, target_name, ax):
    actual_col = "Exp.value" if "Exp.value" in df.columns else "Exp_value"
    predicted_col = "Ensemble"

    metrics = performance_metrics[target_name]

    # Scatter plot
    ax.scatter(df[actual_col], df[predicted_col], color="blue", edgecolor="k", alpha=0.7, s=60)

    # Perfect prediction line
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="Ideal Fit")

    # Labels and title
    ax.set_title(f"{target_name}\nEnsemble Prediction", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    # Add metrics text box (RMSE instead of MSE)
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

plt.suptitle("Actual vs Ensemble Predicted Values with Performance Metrics",
             fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# === Print Performance Summary Table (RMSE instead of MSE) ===
print("Ensemble Model Performance Summary")
print("=" * 50)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 50)
for target, metrics in performance_metrics.items():
    print(f"{target:<25} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R²']:<10.4f}")
print("=" * 50)
