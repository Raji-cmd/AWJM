import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor

# ===== 1️⃣ Prepare AWJM data =====
data = {
    'pressure': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300, 300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                 220, 140, 300, 220, 140, 140, 300],
    'standoff': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1, 3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2, 2.5, 1, 1, 2, 1, 3, 3],
    'traverse': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64, 96, 88, 96, 96, 64, 80, 80, 80, 64, 80, 80, 96, 96, 80, 96, 64,
                 64],
    'mass_flow': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35, 0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45,
                  0.45, 0.55, 0.5, 0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08,
                          4.75, 4.46, 4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8]
}

df = pd.DataFrame(data)
X = df[['pressure', 'standoff', 'traverse', 'mass_flow']].values
y = df['surface_roughness'].values

# ===== 2️⃣ Scale features =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 3️⃣ Define models =====
models = {
    "SVR (RBF Kernel)": SVR(kernel='rbf', C=100, gamma=0.5),
    "Elastic Net": ElasticNet(alpha=0.5, l1_ratio=0.5),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# ===== 4️⃣ Fit models =====
for model in models.values():
    model.fit(X_scaled, y)

# ===== 5️⃣ Prepare grid for plotting =====
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# ===== 6️⃣ Plot predictions =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, model) in zip(axes.ravel(), models.items()):
    # Use first two features for visualization
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(),
    np.full_like(xx.ravel(), X_scaled[:, 2].mean()),  # traverse
    np.full_like(xx.ravel(), X_scaled[:, 3].mean())])  # mass_flow
    Z = Z.reshape(xx.shape)

    contour = ax.contourf(xx, yy, Z, alpha=0.7, cmap='coolwarm')
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    ax.set_title(name)
    fig.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()
