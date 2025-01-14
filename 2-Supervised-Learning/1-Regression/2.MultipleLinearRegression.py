import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath('../Learn-ML'))
from Utils.evalute import evaluate_regression_metrics

# Generate synthetic data
np.random.seed(1)
x = np.random.uniform(0.0, 1.0, size=(100, 2))
y = 30 + 5 * x[:, 0] + 3 * x[:, 1] + 0.5*np.random.normal(0, 1, size=100) # Target variable

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Model parameters
print(f"Intercept (bias): {model.intercept_}")
print(f"Coefficients (weights): {model.coef_}")

# Make predictions using the test data
y_pred = model.predict(x_test)

evaluate_regression_metrics(y_test, y_pred, x_test, model)

# Visualizing the true vs predicted values for testing data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='black', alpha=0.7, label='Predicted vs True')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Perfect Prediction Line")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values (Multiple Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()

# Additional: Display residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='red', edgecolor='black', alpha=0.7)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='black', lw=2, linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.grid(True)
plt.show()