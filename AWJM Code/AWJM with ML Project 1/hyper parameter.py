import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# -------------------------
# Data
# -------------------------
data = {
    'pressure': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300,
                 300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                 220, 140, 300, 220, 140, 140, 300],
    'standoff': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1,
                 3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2,
                 2.5, 1, 1, 2, 1, 3, 3],
    'traverse': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64,
                 96, 88, 96, 96, 64, 80, 80, 80, 64, 80,
                 80, 96, 96, 80, 96, 64, 64],
    'mass_flow': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,
                  0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,
                  0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                          3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                          4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'mrr': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28,
            143.3, 125.7, 106.95, 148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74,
            132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'kerf': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
             1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
             2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
}

df = pd.DataFrame(data)

# Define input features and targets
input_features = ['pressure', 'standoff', 'traverse', 'mass_flow']
targets = ["surface_roughness", "mrr", "kerf"]

# Model configurations for each target (based on your diagrams and table)
model_configs = {
    "surface_roughness": {
        "Ridge": Ridge(alpha=1),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGB": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        "NN": MLPRegressor(hidden_layer_sizes=(6, 4, 3), activation='relu',
                           alpha=0.001, learning_rate_init=0.01,
                           max_iter=1000, random_state=42)
    },
    "mrr": {
        "Ridge": Ridge(alpha=1),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGB": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        "NN": MLPRegressor(hidden_layer_sizes=(8, 5, 4), activation='relu',
                           alpha=0.001, learning_rate_init=0.01,
                           max_iter=1000, random_state=42)
    },
    "kerf": {
        "Ridge": Ridge(alpha=1),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGB": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        "NN": MLPRegressor(hidden_layer_sizes=(6, 4, 2), activation='relu',
                           alpha=0.001, learning_rate_init=0.01,
                           max_iter=1000, random_state=42)
    }
}

results = {}

print("Training on ALL 27 samples...")
print("=" * 50)

for target in targets:
    X = df[input_features]
    y = df[target]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    row = {}
    for name, model in model_configs[target].items():
        # Train on ALL data
        model.fit(X_scaled, y)

        # Predict on the same data (training R²)
        y_pred = model.predict(X_scaled)
        r2 = round(r2_score(y, y_pred), 3)

        row[name] = {"hyperparameters": model.get_params(), "R²": r2}

        print(f"{target} - {name}: R² = {r2}")

    results[target] = row
    print("-" * 30)

# -------------------------
# Convert to DataFrame and save
# -------------------------
flattened_rows = []
for target, model_dict in results.items():
    row = {"Output": target}
    for model_name, metrics in model_dict.items():
        # Clean up hyperparameter display
        hyperparams = metrics["hyperparameters"]
        # Remove lengthy attributes for cleaner display
        for key in ['base_estimator', 'estimator', 'steps', 'verbose', 'warm_start']:
            hyperparams.pop(key, None)
        row[f"{model_name} Hyperparameter"] = str(hyperparams)
        row[f"{model_name} R²"] = metrics["R²"]
    flattened_rows.append(row)

final_excel_df = pd.DataFrame(flattened_rows)


# Save to Excel
final_excel_df.to_excel("Hyperparameter_All_Data.xlsx", index=False)
print("\n" + "=" * 50)
print("Results saved to Hyperparameter_All_Data.xlsx")
print("\nFinal Results Summary:")
print(final_excel_df.to_string(index=False))