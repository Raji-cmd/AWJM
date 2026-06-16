import pandas as pd
import numpy as np

# Your data
data = {
    'pressure_MPa': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300,
                     300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                     220, 140, 300, 220, 140, 140, 300],
    'standoff_distance_mm': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1,
                             3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2,
                             2.5, 1, 1, 2, 1, 3, 3],
    'traverse_speed_mm_min': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64,
                              96, 88, 96, 96, 64, 80, 80, 80, 64, 80,
                              80, 96, 96, 80, 96, 64, 64],
    'mass_flow_rate_kg_min': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,
                              0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,
                              0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness_um': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                             3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                             4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'material_removal_rate_mm3_min': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,
                                      143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,
                                      132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'kerf_angle_deg': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                       1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                       2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
}

df = pd.DataFrame(data)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

# Features and targets
X = df[['pressure_MPa', 'standoff_distance_mm', 'traverse_speed_mm_min', 'mass_flow_rate_kg_min']]
targets = ['surface_roughness_um', 'material_removal_rate_mm3_min', 'kerf_angle_deg']

n_boot = 200  # Number of bootstrap iterations

for target in targets:
    y = df[target]
    preds_boot = np.zeros((n_boot, len(X)))

    # Bootstrap sampling
    for i in range(n_boot):
        X_s, y_s = resample(X, y, replace=True, random_state=42 + i)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_s, y_s)
        preds_boot[i] = model.predict(X)

    # Compute percentiles
    lower = np.percentile(preds_boot, 2.5, axis=0)
    upper = np.percentile(preds_boot, 97.5, axis=0)
    median = np.median(preds_boot, axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(range(len(y)), lower, upper, alpha=0.3, color='gray', label="95% Prediction Interval")
    plt.plot(range(len(y)), y.values, "o-", color='blue', label="Actual", markersize=5, lw=2)
    plt.plot(range(len(y)), median, "s--", color='red', label="Median prediction", markersize=4, lw=2)
    plt.xlabel("Sample Index")
    plt.ylabel(target.replace("_", " ").title())
    plt.title(f"Prediction Intervals (95%) - {target.replace('_', ' ').title()}", fontweight='bold', pad=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
