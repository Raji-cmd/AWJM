import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ===== Updated Kerf Angle Dataset =====
data = {
    'Exp_value': [2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
                  1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
                  2.05,2.43,1.48,1.98,2.08,2.34,1.73],
    'Ridge Regression': [2.02,1.50,1.95,2.27,1.80,2.23,1.91,2.10,1.73,1.64,
                         1.96,2.00,2.55,1.68,1.95,2.10,1.95,1.91,1.51,1.88,
                         1.99,2.40,1.52,1.95,2.12,2.38,1.79],
    'RandomForest': [2.13,1.51,1.98,2.22,1.73,2.27,1.94,2.10,1.77,1.65,
                     1.82,2.01,2.50,1.64,1.93,2.11,1.98,1.95,1.56,1.95,
                     2.03,2.42,1.56,1.98,2.10,2.34,1.71],
    'XGBoost': [2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
                1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
                2.05,2.43,1.48,1.98,2.08,2.34,1.73],
    'NeuralNet': [1.96,1.46,1.86,2.32,1.81,2.31,1.80,2.11,1.65,1.65,
                  1.90,1.91,2.60,1.70,1.93,2.08,1.86,1.83,1.60,1.79,
                  1.91,2.49,1.55,1.86,2.11,2.37,1.81]
}

df = pd.DataFrame(data)

# Prepare input features (model predictions as meta-features)
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
plt.title("Learning Curve - Ensemble Kerf Angle", fontsize=16, fontweight='bold')
plt.legend(loc="best", fontsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
