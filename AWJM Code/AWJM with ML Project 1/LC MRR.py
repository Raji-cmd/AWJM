import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ===== New MRR Dataset =====
data = {
    'Exp_value': [119.98,126.36,122.40,111.57,126.63,97.63,121.60,109.81,137.08,117.28,
                  143.30,125.70,106.95,148.41,102.02,115.20,122.40,119.69,145.92,131.74,
                  132.71,99.70,140.98,122.40,104.26,103.66,139.85],
    'Ridge Regression': [120.20,126.70,122.19,117.72,130.56,94.96,121.22,113.82,134.08,126.66,
                         141.46,123.17,109.76,149.42,102.93,114.27,122.19,119.47,145.52,124.18,
                         124.92,98.87,138.53,122.19,106.83,105.86,137.55],
    'RandomForest': [120.80,130.02,122.64,110.71,127.61,98.92,121.74,109.21,132.05,122.34,
                     141.74,124.28,106.82,146.05,102.65,107.63,122.64,121.00,144.99,129.38,
                     128.66,100.41,137.34,122.64,104.17,105.10,140.65],
    'XGBoost': [119.98,126.36,122.40,111.57,126.63,97.63,121.60,109.81,137.08,117.28,
                143.30,125.70,106.95,148.41,102.02,115.20,122.40,119.69,145.92,131.74,
                132.71,99.70,140.98,122.40,104.26,103.66,139.85],
    'NeuralNet': [113.10,129.65,115.53,117.42,131.27,100.46,114.50,116.93,130.43,126.20,
                  147.24,116.57,111.83,155.28,105.86,105.60,115.53,111.80,150.06,117.96,
                  119.27,103.35,145.04,115.53,111.00,107.69,145.27]
}

df = pd.DataFrame(data)

# Prepare input features (using model predictions as meta-features)
X = df[['Ridge Regression', 'RandomForest', 'XGBoost', 'NeuralNet']].values
y = df['Exp_value'].values

# Define base regressors
ridge = Ridge()
rf = RandomForestRegressor(random_state=0)
xgb = XGBRegressor(random_state=0, verbosity=0)
nn = MLPRegressor(random_state=0, max_iter=1000)

# Ensemble (Voting Regressor)
ensemble = VotingRegressor([('ridge', ridge), ('rf', rf), ('xgb', xgb), ('nn', nn)])

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    ensemble, X, y, cv=5, scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.2, 1.0, 5)
)

# Mean and std
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot
plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation Score')

# Confidence intervals
plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                 train_scores_mean+train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                 test_scores_mean+test_scores_std, alpha=0.1, color='g')

plt.xlabel("Training Examples", fontsize=14, fontweight='bold')
plt.ylabel("Mean Squared Error", fontsize=14, fontweight='bold')
plt.title("Learning Curve - Ensemble MRR", fontsize=16, fontweight='bold')
plt.legend(loc="best", fontsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
