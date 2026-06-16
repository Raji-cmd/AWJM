import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------
# 1. Data: Study Hours vs Marks
# ----------------------
X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)  # Hours studied
y = np.array([35, 40, 50, 55, 65, 70, 75])         # Marks obtained

# ----------------------
# 2. Build Linear Regression Model
# ----------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------
# 3. Predictions
# ----------------------
y_pred = model.predict(X)
new_hours = 8
predicted_marks = model.predict([[new_hours]])

# ----------------------
# 4. Goodness of Fit Metrics
# ----------------------
r2 = model.score(X, y)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# ----------------------
# 5. Visualization
# ----------------------
plt.scatter(X, y, color='blue', label='Actual Marks')
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.scatter(new_hours, predicted_marks, color='green', label='Predicted Point')

# Add metrics inside the plot
plt.text(1, 30, f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}",
         fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

plt.xlabel('Study Hours')
plt.ylabel('Marks Obtained')
plt.title('Student Marks Prediction - Goodness of Fit')
plt.legend()
plt.show()
