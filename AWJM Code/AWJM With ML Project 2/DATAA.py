import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Step 1: Load Data ===
data = {
    'pressure':[220, 260, 220, 140, 300, 140, 220, 140, 280, 300,
                300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                220, 140, 300, 220, 140, 140, 300],
    'standoff':[2, 1, 2, 3, 1, 1, 2, 3, 2, 1,
                3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2,
                2.5, 1, 1, 2, 1, 3, 3],
    'traverse':[80, 64, 80, 96, 96, 64, 72, 64, 80, 64,
                96, 88, 96, 96, 64, 80, 80, 80, 64, 80,
                80, 96, 96, 80, 96, 64, 64],
    'mass_flow':[0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,
                 0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,
                 0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.5,
                         3.97,4.63,5.56,3.62,4.08,4.75,4.46,4.41,3.42,4.27,
                         4.67,5.36,3.29,4.46,4.53,5.23,3.8],
    'mrr':[119.98,126.36,122.4,111.57,126.63,97.63,121.6,109.81,137.08,117.28,
           143.3,125.7,106.95,148.41,102.02,115.2,122.4,119.69,145.92,131.74,
           132.71,99.7,140.98,122.4,104.26,103.66,139.85],
    'kerf':[2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
            1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
            2.05,2.43,1.48,1.98,2.08,2.34,1.73]
}

df = pd.DataFrame(data)

# === Step 2: Feature Scaling ===
X = df[['pressure','standoff','traverse','mass_flow']].values
X_scaled = StandardScaler().fit_transform(X)

# === Step 3: Train Models and Get Predictions ===
def run_models(y):
    preds = {}

    preds["Ridge"] = Ridge(alpha=1.0, random_state=42).fit(X_scaled, y).predict(X_scaled)
    preds["RandomForest"] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_scaled, y).predict(X_scaled)
    preds["SVR"] = SVR(kernel='rbf', C=10, gamma='scale').fit(X_scaled, y).predict(X_scaled)
    preds["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X_scaled, y).predict(X_scaled)
    preds["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0).fit(X_scaled, y).predict(X_scaled)
    preds["NeuralNet"] = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=2000, random_state=42).fit(X_scaled, y).predict(X_scaled)

    return preds

# === Step 4: Build DataFrame with Predictions and Metrics ===
def make_df(y):
    preds = run_models(y)
    out = pd.DataFrame({"Exp.value": y})
    for name, val in preds.items():
        out[name] = val

    # --- Performance Metrics ---
    metrics = []
    for name, val in preds.items():
        r2 = r2_score(y, val)
        rmse = np.sqrt(mean_squared_error(y, val))
        mae = mean_absolute_error(y, val)
        metrics.append([name, r2, rmse, mae])

    metrics_df = pd.DataFrame(metrics, columns=["Model", "R²", "RMSE", "MAE"])
    return out, metrics_df

# === Step 5: Generate for Each Output ===
df_surface, metrics_surface = make_df(df['surface_roughness'].values)
df_mrr, metrics_mrr = make_df(df['mrr'].values)
df_kerf, metrics_kerf = make_df(df['kerf'].values)

# === Step 6: Save to Excel ===
with pd.ExcelWriter("Data_All_6Models_with_Metrics.xlsx") as writer:
    df_surface.round(2).to_excel(writer, sheet_name="Surface_Roughness", index=False)
    metrics_surface.round(4).to_excel(writer, sheet_name="Surface_Roughness_Metrics", index=False)

    df_mrr.round(2).to_excel(writer, sheet_name="MRR", index=False)
    metrics_mrr.round(4).to_excel(writer, sheet_name="MRR_Metrics", index=False)

    df_kerf.round(2).to_excel(writer, sheet_name="Kerf", index=False)
    metrics_kerf.round(4).to_excel(writer, sheet_name="Kerf_Metrics", index=False)

print("✅ Saved: Data_All_6Models_with_Metrics.xlsx (Includes R², RMSE, MAE for each model)")
