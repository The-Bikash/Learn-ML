import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(1)

# Step 1: Generate custom synthetic data
x = np.random.uniform(0.0, 1.0, size=(100, 2))  # 100 samples, 2 features
y = 30 + 5 * x[:, 0] + 3 * x[:, 1] + 0.5 * np.random.normal(0, 1, size=100)  # Target variable with noise

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Step 3: Apply Ridge Regression
ridge = Ridge(alpha=1.0)  # Regularization strength
ridge.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = ridge.predict(X_test)

# Step 5: Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Step 6: Visualize the results
# 3D Visualization of the data points and regression plane
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the training data
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='blue', label='Training data', alpha=0.5)

# Plot the test data
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='green', label='Test data', alpha=0.5)

# Create a meshgrid for the 2D input space
x_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 10)
y_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10)
xx, yy = np.meshgrid(x_range, y_range)
zz = ridge.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot the regression plane
ax.plot_surface(xx, yy, zz, color='red', alpha=0.5, label='Ridge regression plane')

# Labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Ridge Regression with Custom Synthetic Data')

# Display legend
ax.legend()

# Show plot
plt.show()

# Visualizing the learned coefficients
print(f"Ridge Regression Coefficients: {ridge.coef_}")
