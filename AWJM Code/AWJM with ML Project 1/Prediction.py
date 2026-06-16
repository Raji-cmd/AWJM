import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Your data
data = {
    'pressure': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300, 300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                 220, 140, 300, 220, 140, 140, 300],
    'standoff': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1, 3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2, 2.5, 1, 1, 2, 1, 3, 3],
    'traverse': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64, 96, 88, 96, 96, 64, 80, 80, 80, 64, 80, 80, 96, 96, 80, 96, 64,
                 64],
    'mass_flow': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35, 0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45,
                  0.45, 0.55, 0.5, 0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08, 4.75,
                          4.46, 4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'mrr': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95, 148.41,
            102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66, 139.85],
    'kerf': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59, 1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96,
             1.52, 1.93, 2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
}

# Create DataFrame
df = pd.DataFrame(data)
exp_nos = list(range(1, len(df) + 1))

# Input features and target variables
X = df[['pressure', 'standoff', 'traverse', 'mass_flow']]
targets = {
    'Surface Roughness': df['surface_roughness'],
    'Material Removal Rate': df['mrr'],
    'Kerf Angle': df['kerf']
}

# Initialize models
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}


# Function to train models and get predictions
def get_predictions(X, y, test_size=0.2):
    predictions = {}
    scores = {}
    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if name == 'Neural Network':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        test_indices = X_test.index
        full_pred = np.zeros(len(X))
        full_pred[test_indices] = y_pred
        train_indices = X_train.index
        full_pred[train_indices] = y.iloc[train_indices].values
        predictions[name] = full_pred
        scores[name] = r2_score(y_test, y_pred)
    return predictions, scores


# Colors
colors = ['red', 'blue', 'green', 'orange']
model_names = list(models.keys())

from sklearn.metrics import mean_absolute_error


# Function to calculate metrics
def calculate_metrics(y_true, predictions):
    metrics = {}
    for name, pred in predictions.items():
        mse = mean_squared_error(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)
        metrics[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}
    return metrics


# Plot & analysis for each target separately
for target_name, y in targets.items():
    predictions, _ = get_predictions(X, y)
    metrics = calculate_metrics(y, predictions)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(exp_nos, y, 'ko-', linewidth=3, markersize=8, label='Exp Value', markerfacecolor='black')
    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(exp_nos, pred, 's-', color=colors[i], markersize=5, label=name, alpha=0.8, linewidth=2)

    plt.title(f'{target_name} Model Predictions', fontweight='bold', fontsize=18)
    plt.xlabel('Experiment Number',fontsize=18, fontweight='bold')
    plt.ylabel(target_name, fontsize=18,fontweight='bold')

    # Legend outside
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9, markerscale=0.7, frameon=True, borderpad=0.3)

    plt.grid(True, alpha=0.3)
    plt.xticks(exp_nos[::3])
    plt.tight_layout()
    plt.show()

    # Print metrics
    print(f"\n{target_name} Model Performance Metrics:")
    print(f"{'Model':<20} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    for model, metric in metrics.items():
        print(f"{model:<20} {metric['MSE']:>10.4f} {metric['MAE']:>10.4f} {metric['R²']:>10.4f}")
