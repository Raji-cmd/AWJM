import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# === Step 1: Load Data ===
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
    'material_removal_rate_mm3_min': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63,
                                      121.6, 109.81, 137.08, 117.28, 143.3, 125.7,
                                      106.95, 148.41, 102.02, 115.2, 122.4, 119.69,
                                      145.92, 131.74, 132.71, 99.7, 140.98, 122.4,
                                      104.26, 103.66, 139.85],
    'Kerf':[2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
            1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
            2.05,2.43,1.48,1.98,2.08,2.34,1.73]
}

df = pd.DataFrame(data)
df.columns = ['pressure','standoff','traverse','mass_flow','surface_roughness','mrr','kerf']
df = df.astype(float)

# === Step 2: Features and Scaling ===
X = df[['pressure','standoff','traverse','mass_flow']].values
X_scaled = StandardScaler().fit_transform(X)

# === Step 3: Train models and compute ensemble ===
def run_models_weighted(y, weights=[0.15, 0.25, 0.30, 0.30]):
    preds = {}

    # Base Models
    preds["Ridge"] = Ridge().fit(X_scaled, y).predict(X_scaled)
    preds["RandomForest"] = RandomForestRegressor().fit(X_scaled, y).predict(X_scaled)
    preds["XGBoost"] = xgb.XGBRegressor(objective='reg:squarederror').fit(X_scaled, y).predict(X_scaled)
    preds["NeuralNet"] = MLPRegressor(max_iter=2000).fit(X_scaled, y).predict(X_scaled)

    # Weighted Ensemble
    base_preds = np.column_stack([preds["Ridge"], preds["RandomForest"], preds["XGBoost"], preds["NeuralNet"]])
    preds["Ensemble"] = base_preds @ np.array(weights)

    return preds

# === Step 4: Compute metrics ===
def compute_metrics(y_true, y_pred):
    return {
        "R²": round(r2_score(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3)
    }

# === Step 5: Create results DataFrame ===
def make_df_weighted(y, target_name):
    preds = run_models_weighted(y)
    out = pd.DataFrame({"Exp.value": y})
    metrics_list = []

    # Add predictions
    for name, val in preds.items():
        out[name] = val
        metrics = compute_metrics(y, val)
        metrics_list.append([name, metrics["R²"], metrics["RMSE"], metrics["MAE"]])

    # Add metrics summary
    metrics_df = pd.DataFrame(metrics_list, columns=["Model", "R²", "RMSE", "MAE"])
    return out.round(3), metrics_df

# === Step 6: Run for all targets ===
df_surface, metrics_surface = make_df_weighted(df['surface_roughness'].values, "Surface_Roughness")
df_mrr, metrics_mrr = make_df_weighted(df['mrr'].values, "MRR")
df_kerf, metrics_kerf = make_df_weighted(df['kerf'].values, "Kerf")

# === Step 7: Save all to Excel ===
with pd.ExcelWriter("Weighted_Ensemble_Data.xlsx") as writer:
    df_surface.to_excel(writer, sheet_name="Surface_Roughness", index=False)
    metrics_surface.to_excel(writer, sheet_name="Surface_Roughness_Metrics", index=False)

    df_mrr.to_excel(writer, sheet_name="MRR", index=False)
    metrics_mrr.to_excel(writer, sheet_name="MRR_Metrics", index=False)

    df_kerf.to_excel(writer, sheet_name="Kerf", index=False)
    metrics_kerf.to_excel(writer, sheet_name="Kerf_Metrics", index=False)

print("✅ Saved: Weighted_Ensemble_Data.xlsx with metrics (R², RMSE, MAE)")
