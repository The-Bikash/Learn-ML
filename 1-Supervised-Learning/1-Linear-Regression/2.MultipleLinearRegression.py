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
x = np.random.uniform(0, 1.0, size=(100, 2))
y = 30 + 5 * x[:, 0] + 3 * x[:, 1] + np.random.normal(0, 1.0, size=100) # Target variable

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Model parameters
print(f"Intercept (bias): {model.intercept_}")
print(f"Coefficients (weights): {model.coef_}")