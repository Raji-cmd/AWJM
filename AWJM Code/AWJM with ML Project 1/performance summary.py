from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# -------------------------
# Actual (experimental) values
# -------------------------
surface_actual = np.array([
    4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.50,
    3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
    4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.80
])

mrr_actual = np.array([
    119.98, 126.36, 122.40, 111.57, 126.63, 97.63, 121.60, 109.81, 137.08,
    117.28, 143.30, 125.70, 106.95, 148.41, 102.02, 115.20, 122.40, 119.69,
    145.92, 131.74, 132.71, 99.70, 140.98, 122.40, 104.26, 103.66, 139.85
])

kerf_actual = np.array([
    2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
    1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
    2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73
])

# -------------------------
# Predicted (ensemble) values
# -------------------------
surface_pred = np.array([
    4.65, 3.25, 4.43, 5.02, 3.89, 5.03, 4.34, 4.65, 3.90, 3.58,
    4.05, 4.52, 5.56, 3.64, 4.17, 4.71, 4.43, 4.34, 3.45, 4.25,
    4.57, 5.36, 3.34, 4.43, 4.58, 5.25, 3.83
])

mrr_pred = np.array([
    117.92, 127.71, 120.20, 114.28, 129.46, 98.73, 119.33, 111.19,
    133.08, 122.76, 143.58, 122.12, 109.42, 150.70, 104.27, 110.69,
    120.20, 117.54, 147.84, 125.59, 126.23, 101.32, 141.08, 120.20,
    105.96, 106.23, 139.99
])

kerf_pred = np.array([
    2.065, 1.494, 1.947, 2.259, 1.76, 2.25, 1.904, 2.105, 1.774, 1.632,
    1.864, 1.997, 2.563, 1.663, 1.907, 2.09, 1.947, 1.913, 1.538, 1.892,
    1.99, 2.424, 1.51, 1.947, 2.096, 2.369, 1.751
])

# -------------------------
# Metric calculation function
# -------------------------
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# -------------------------
# Compute metrics for each target
# -------------------------
surface_metrics = get_metrics(surface_actual, surface_pred)
mrr_metrics = get_metrics(mrr_actual, mrr_pred)
kerf_metrics = get_metrics(kerf_actual, kerf_pred)

# -------------------------
# Display results
# -------------------------
print("Ensemble Performance Metrics (RMSE, MAE, R²):")
print(f"{'Target':<25}{'RMSE':>10}{'MAE':>10}{'R²':>10}")
print(f"{'Surface Roughness':<25}{surface_metrics[0]:>10.4f}{surface_metrics[1]:>10.4f}{surface_metrics[2]:>10.4f}")
print(f"{'Material Removal Rate':<25}{mrr_metrics[0]:>10.4f}{mrr_metrics[1]:>10.4f}{mrr_metrics[2]:>10.4f}")
print(f"{'Kerf Angle':<25}{kerf_metrics[0]:>10.4f}{kerf_metrics[1]:>10.4f}{kerf_metrics[2]:>10.4f}")
