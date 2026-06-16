import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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


# === Create DataFrames ===
datasets = {
    "Surface Roughness": pd.DataFrame(surface_data),
    "Material Removal Rate": pd.DataFrame(mrr_data),
    "Kerf Angle": pd.DataFrame(kerf_data)
}

# Color scheme
colors = ['#1f77b4', '#ff7f0e']  # Blue: Actual, Orange: Predicted

# === Function: Plot Density & Summary ===
# === Function: Plot Density & Summary ===
def plot_and_summarize(df, metric_name):
    plt.figure(figsize=(6,4))
    sns.kdeplot(df['Exp.value'], label='Actual', color=colors[0], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(df['Ensemble'], label='Predicted', color=colors[1], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)

    # Title and axis labels with fontsize + fontweight
    plt.title(f"{metric_name}: Actual vs Predicted Density", fontsize=14, fontweight='bold')
    plt.xlabel(metric_name, fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')

    # Legend styling
    plt.legend(fontsize=10, title_fontsize=11)

    plt.grid(True, alpha=0.3)
    plt.show()


    # Statistical Summary
    print(f"\n=== {metric_name} Statistical Summary ===")
    print(f"Actual   - Mean: {df['Exp.value'].mean():.4f}, Std: {df['Exp.value'].std():.4f}")
    print(f"Predicted- Mean: {df['Ensemble'].mean():.4f}, Std: {df['Ensemble'].std():.4f}")

    # KS Test
    ks_stat, p_value = stats.ks_2samp(df['Exp.value'], df['Ensemble'])
    print(f"Kolmogorov-Smirnov Test: Stat={ks_stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("Distributions are statistically similar (p > 0.05)")
    else:
        print("Distributions are statistically different (p ≤ 0.05)")


# === Apply Function to All Datasets ===
for name, df in datasets.items():
    plot_and_summarize(df, name)
