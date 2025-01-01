from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_regression_metrics(y_true, y_pred, X=None, model=None, cv_folds=5):
    """
    Evaluates a set of regression metrics given true and predicted values.
    
    Parameters:
    - y_true: True target values
    - y_pred: Predicted values from the model
    - X: Feature data (optional, required for cross-validation)
    - model: Trained model (optional, required for cross-validation)
    - cv_folds: Number of folds for cross-validation (default is 5)
    
    Returns:
    - A dictionary with regression performance metrics
    """
    # Calculate R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Adjusted R-squared
    n = len(y_true)  # Number of observations
    p = X.shape[1] if X is not None else 1  # Number of predictors
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # Adjusted R-squared formula
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Median Absolute Error
    medae = median_absolute_error(y_true, y_pred)
    
    # Explained Variance Score
    evs = explained_variance_score(y_true, y_pred)
    
    # Cross-validation R-squared (if model and X are provided)
    cv_r2 = None
    if X is not None and model is not None:
        cv_r2 = cross_val_score(model, X, y_true, cv=cv_folds, scoring='r2').mean()

    # Return results in a dictionary
    print(
        {
        "R-squared": r2,
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "Adjusted R-squared": r2_adj,
        "Mean Absolute Percentage Error (MAPE)": mape,
        "Median Absolute Error": medae,
        "Explained Variance Score": evs,
        "Cross-validation R-squared (CV)": cv_r2,
        }
    )
