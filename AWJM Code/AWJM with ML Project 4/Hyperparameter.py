import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import tensorflow as tf



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

# -------------------------
# Define input features and targets
# -------------------------
input_features = ['pressure', 'standoff', 'traverse', 'mass_flow']
targets = ["surface_roughness", "mrr", "kerf"]

# -------------------------
# Neural Network builder
# -------------------------
def build_nn():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# -------------------------
# Model configurations
# -------------------------
def get_model_configs():
    return {
        "Ridge": Ridge(alpha=1),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42),
        "SVR": SVR(C=10, epsilon=0.1, kernel='rbf'),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.9,
            colsample_bytree=0.8, random_state=42
        ),
        "NeuralNet": KerasRegressor(build_fn=build_nn, epochs=200, batch_size=8, verbose=0)
    }

# -------------------------
# Training
# -------------------------
results = {}
print("Training on ALL 27 samples...")
print("=" * 50)

for target in targets:
    X = df[input_features]
    y = df[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_configs = get_model_configs()
    row = {}

    for name, model in model_configs.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = round(r2_score(y, y_pred), 3)
        try:
            hyperparams = model.get_params()
        except:
            hyperparams = {"model": "Keras Sequential"}
        row[name] = {"hyperparameters": hyperparams, "R²": r2}
        print(f"{target} - {name}: R² = {r2}")

    results[target] = row
    print("-" * 40)

# -------------------------
# Convert to DataFrame and save
# -------------------------
flattened_rows = []
for target, model_dict in results.items():
    row = {"Output": target}
    for model_name, metrics in model_dict.items():
        hyperparams = metrics["hyperparameters"]
        for key in ['base_estimator', 'estimator', 'steps', 'verbose', 'warm_start']:
            hyperparams.pop(key, None)
        row[f"{model_name} Hyperparameter"] = str(hyperparams)
        row[f"{model_name} R²"] = metrics["R²"]
    flattened_rows.append(row)

final_excel_df = pd.DataFrame(flattened_rows)
final_excel_df.to_excel("Hyperparameter_All_Data_With_XGB_NN.xlsx", index=False)

print("\n✅ Results saved to 'Hyperparameter_All_Data_With_XGB_NN.xlsx'")
print("\nFinal Summary:")
print(final_excel_df.to_string(index=False))
