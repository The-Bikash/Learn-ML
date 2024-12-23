from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('../Learn-ML'))
from Utils.evalute import evaluate_regression_metrics


# Generate synthetic data
np.random.seed(0)
x = 3 * np.random.uniform(0.0, 1.0, size=(1000, 1))
y = 4 + 3 * x + 0.5 * np.random.normal(0.0, 1.0, size=(1000, 1)) # y = 4 + 3x + noise

#split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Model Parameters
print(f"Intercept (bias): {model.intercept_[0]}")
print(f"Slop (weight): {model.coef_[0][0]}")

# Predictions
y_pred = model.predict(x_test)

#Evalute the Model
metrics = evaluate_regression_metrics(y_test, y_pred, x_test, model)
for metric, value in metrics.items():
    print(f"{metric}: {value}")


# Plot results

# Plot the regression line for the entire range of x
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_range = model.predict(x_range)
plt.plot(x_range, y_range, color='green', label='Regression Line')

# Plot test data (red) and predictions (yellow)
plt.scatter(x_train, y_train, color='blue', label='Training Data', s=1)
plt.scatter(x_test, y_test, color='red', label='Test Data', s=1)
plt.scatter(x_test, y_pred, color='yellow', label='Predictions', s=1)

# Labeling the plot
plt.title('Train-Test Split and Model Fit')
plt.xlabel('Feature (x)')
plt.ylabel('Target (y)')
plt.legend()

# Show the plot
plt.show()
