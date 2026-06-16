import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

surface_data = {
 "Exp.value": [4.75,3.09,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.50,
                  3.97,4.63,5.56,3.62,4.08,4.75,4.46,4.41,3.42,4.27,
                  4.67,5.36,3.29,4.46,4.53,5.23,3.80],
    "Ridge": [4.49,3.37,4.34,5.03,3.98,4.96,4.26,4.70,3.85,3.65,
              4.33,4.42,5.64,3.72,4.35,4.67,4.34,4.25,3.38,4.19,
              4.43,5.30,3.37,4.34,4.69,5.31,3.99],
    "RandomForest": [4.85,3.44,4.47,4.86,3.81,5.01,4.37,4.66,3.94,3.61,
                     3.93,4.56,5.44,3.57,4.26,4.67,4.47,4.42,3.47,4.33,
                     4.59,5.29,3.39,4.47,4.53,5.19,3.77],
    "SVR": [4.73,3.19,4.51,4.92,3.94,4.94,4.32,4.51,4.04,3.60,
            4.07,4.61,5.46,3.72,4.04,4.85,4.51,4.32,3.52,4.20,
            4.62,5.26,3.39,4.51,4.43,5.13,3.90],
    "ElasticNet": [4.46,3.60,4.34,4.90,3.99,4.95,4.29,4.69,3.90,3.79,
                   4.21,4.39,5.37,3.73,4.47,4.63,4.34,4.29,3.53,4.22,
                   4.39,5.16,3.52,4.34,4.68,5.16,4.00],
    "XGBoost": [4.75,3.10,4.46,5.02,3.84,5.04,4.38,4.61,4.14,3.50,
                3.97,4.63,5.55,3.62,4.08,4.75,4.46,4.41,3.42,4.27,
                4.67,5.36,3.29,4.46,4.53,5.23,3.80],
    "NeuralNet": [4.65,3.11,4.47,5.00,3.96,5.04,4.40,4.66,3.99,3.56,
                  4.10,4.48,5.59,3.64,4.13,4.80,4.47,4.23,3.44,4.32,
                  4.50,5.34,3.24,4.47,4.60,5.23,3.77],
    "Ensemble": [4.67,3.27,4.44,4.95,3.92,4.99,4.34,4.63,4.00,3.61,
                 4.08,4.53,5.51,3.66,4.20,4.74,4.44,4.32,3.47,4.26,
                 4.55,5.29,3.36,4.44,4.56,5.20,3.86]

}

mrr_data = {
    "Exp.value": [119.98,126.36,122.40,111.57,126.63,97.63,121.60,109.81,137.08,117.28,
                  143.30,125.70,106.95,148.41,102.02,115.20,122.40,119.69,145.92,131.74,
                  132.71,99.70,140.98,122.40,104.26,103.66,139.85],
    "Ridge": [120.20,126.70,122.19,117.72,130.56,94.96,121.22,113.82,134.08,126.66,
              141.46,123.17,109.76,149.42,102.93,114.27,122.19,119.47,145.52,124.18,
              124.92,98.87,138.53,122.19,106.83,105.86,137.55],
    "RandomForest": [120.95,129.90,122.37,110.20,127.90,99.32,121.86,109.28,132.74,121.52,
                     141.69,124.35,107.29,145.93,103.14,109.61,122.37,120.24,144.43,128.74,
                     128.68,100.53,136.81,122.37,104.26,105.85,139.48],
    "SVR": [120.08,126.26,124.14,111.67,126.53,101.68,121.50,109.91,136.98,117.38,
            133.90,125.60,107.57,136.78,107.07,113.51,124.14,119.79,134.81,127.03,
            127.19,104.16,133.05,124.14,107.34,105.90,131.03],
    "ElasticNet": [120.26,126.65,122.19,117.66,130.47,95.53,121.25,113.91,133.89,126.72,
                   141.12,123.13,109.93,148.85,103.26,114.39,122.19,119.53,145.10,124.12,
                   124.85,99.28,138.20,122.19,107.01,106.18,137.37],
    "XGBoost": [120.02,126.36,122.42,111.57,126.63,97.77,121.63,109.81,137.04,117.30,
                143.28,125.65,106.95,148.27,102.05,115.20,122.42,119.72,145.91,131.69,
                132.66,99.73,140.97,122.42,104.27,103.69,139.85],
    "NeuralNet": [119.84,126.51,124.80,111.83,126.69,97.71,120.72,110.02,136.30,117.53,
                  143.31,125.71,107.02,148.59,102.51,113.16,124.80,117.29,145.95,129.53,
                  130.99,99.67,141.18,124.80,104.60,103.76,140.07],
    "Ensemble": [120.19,126.98,123.18,112.97,127.78,98.15,121.36,110.81,135.47,120.34,
                 140.67,124.83,107.87,145.89,103.58,113.40,123.18,119.27,143.32,128.00,
                 128.69,100.57,138.14,123.18,105.61,105.06,137.47]
}

kerf_data = {
 "Exp.value": [2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
                  1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
                  2.05,2.43,1.48,1.98,2.08,2.34,1.73],
    "Ridge": [2.02,1.50,1.95,2.27,1.80,2.23,1.91,2.10,1.73,1.64,
              1.96,2.00,2.55,1.68,1.95,2.10,1.95,1.91,1.51,1.88,
              1.99,2.40,1.52,1.95,2.12,2.38,1.79],
    "RandomForest": [2.14,1.54,1.98,2.19,1.75,2.27,1.95,2.10,1.76,1.64,
                     1.83,2.02,2.49,1.61,1.95,2.11,1.98,1.96,1.55,1.95,
                     2.02,2.41,1.54,1.98,2.08,2.34,1.71],
    "SVR": [2.07,1.54,1.96,2.15,1.85,2.15,1.89,1.99,1.73,1.69,
            1.96,2.02,2.47,1.74,1.86,2.13,1.96,1.89,1.62,1.84,
            2.02,2.33,1.58,1.96,1.98,2.24,1.83],
    "ElasticNet": [1.99,1.73,1.95,2.14,1.81,2.22,1.94,2.10,1.78,1.77,
                   1.84,1.96,2.29,1.69,2.07,2.07,1.95,1.95,1.65,1.92,
                   1.96,2.26,1.66,1.95,2.11,2.25,1.80],
    "XGBoost": [2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
                1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
                2.05,2.43,1.48,1.98,2.08,2.34,1.73],
    "NeuralNet": [1.97,1.42,1.95,2.22,1.85,2.25,1.93,2.12,1.70,1.65,
                  1.93,1.91,2.67,1.70,1.92,2.08,1.95,1.94,1.57,1.87,
                  1.96,2.40,1.46,1.95,2.13,2.37,1.71],
    "Ensemble": [2.05,1.52,1.96,2.20,1.80,2.23,1.93,2.08,1.76,1.66,
                 1.89,1.99,2.51,1.68,1.93,2.11,1.96,1.94,1.57,1.90,
                 2.00,2.37,1.54,1.96,2.08,2.32,1.76]
}



# Assume surface_data, mrr_data, kerf_data are already defined

def plot_learning_curve(data, title, ylabel):
    # Features: all model predictions except Ensemble and Exp.value
    X = np.column_stack([data['Ridge'], data['RandomForest'], data['ElasticNet'],
                         data['SVR'], data['XGBoost'], data['NeuralNet']])
    y = np.array(data['Exp.value'])

    # Base models
    ridge = Ridge()
    rf = RandomForestRegressor(random_state=0)
    enet = ElasticNet()
    svr = SVR()
    xgb = XGBRegressor(objective='reg:squarederror', random_state=0)
    nn = MLPRegressor(random_state=0, max_iter=1000)

    # Voting Regressor ensemble including XGBoost and NeuralNet
    ensemble = VotingRegressor([
        ('ridge', ridge),
        ('rf', rf),
        ('enet', enet),
        ('svr', svr),
        ('xgb', xgb),
        ('nn', nn)
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        ensemble, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation Score')
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std, alpha=0.1, color='g')
    plt.xlabel("Training Examples", fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(f"Learning Curve - Ensemble {title}", fontsize=16, fontweight='bold')
    plt.legend(loc="best", fontsize=12)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

# Run for all datasets
plot_learning_curve(surface_data, "Surface Roughness", "Mean Squared Error")
plot_learning_curve(mrr_data, "Material Removal Rate", "Mean Squared Error")
plot_learning_curve(kerf_data, "Kerf Angle", "Mean Squared Error")
