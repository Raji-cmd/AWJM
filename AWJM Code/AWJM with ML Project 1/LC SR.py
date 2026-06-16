import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# ===== New Dataset =====
data = {
    'Exp_value':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.50,3.97,4.63,5.56,3.62,4.08,
                 4.75,4.46,4.41,3.42,4.27,4.67,5.36,3.29,4.46,4.53,5.23,3.80],
    'Ridge':[4.49,3.37,4.34,5.03,3.98,4.96,4.26,4.70,3.85,3.65,4.33,4.42,5.64,3.72,4.35,
             4.67,4.34,4.25,3.38,4.19,4.43,5.30,3.37,4.34,4.69,5.31,3.99],
    'RandomForest':[4.83,3.39,4.46,4.90,3.78,5.06,4.38,4.61,3.91,3.62,3.93,4.57,5.45,3.56,4.24,
                    4.68,4.46,4.40,3.45,4.35,4.60,5.30,3.39,4.46,4.60,5.20,3.79],
    'XGBoost':[4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.50,3.97,4.63,5.56,3.62,4.08,
               4.75,4.46,4.41,3.42,4.27,4.67,5.36,3.29,4.46,4.53,5.23,3.80],
    'NeuralNet':[4.47,3.17,4.40,5.07,3.93,5.11,4.30,4.68,3.92,3.56,4.05,4.40,5.63,3.71,4.08,
                 4.70,4.40,4.27,3.43,4.19,4.48,5.43,3.35,4.40,4.57,5.26,3.89]
}

# Features and target
X = np.column_stack([data['Ridge'], data['RandomForest'], data['XGBoost'], data['NeuralNet']])
y = np.array(data['Exp_value'])

# Base regressors
ridge = Ridge()
rf = RandomForestRegressor(random_state=0)
xgb = XGBRegressor(random_state=0, verbosity=0)
nn = MLPRegressor(random_state=0, max_iter=1000)

# Voting ensemble
ensemble = VotingRegressor([('ridge', ridge), ('rf', rf), ('xgb', xgb), ('nn', nn)])

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    ensemble, X, y, cv=5, scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
)

# Convert to positive MSE
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot
plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation Score')

plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                 train_scores_mean+train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                 test_scores_mean+test_scores_std, alpha=0.1, color='g')

plt.xlabel("Training Examples", fontsize=14, fontweight='bold')
plt.ylabel("Mean Squared Error", fontsize=14, fontweight='bold')
plt.title("Learning Curve - Ensemble Surface Roughness", fontsize=16, fontweight='bold')
plt.legend(loc="best", fontsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
