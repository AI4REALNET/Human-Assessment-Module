import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

def RANSAC_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object]:
    """
    Training of RANSAC Regression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    A robust regression method that iteratively fits a model while ignoring outliers.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - RANSACRegressor (object): Trained RANSACRegressor model from `sklearn.linear_model`.
    """

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define base estimator (Linear Regression)
    base_estimator = LinearRegression()

    # Define RANSAC model
    ransac = RANSACRegressor(estimator=base_estimator, random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'min_samples': [0.1, 0.3, 0.5],  # Percentage of data samples to fit
        'residual_threshold': [1.0, 2.0, 5.0]  # Threshold to determine inliers
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        ransac, 
        param_grid, 
        cv=2, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_ransac = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions
    y_pred = best_ransac.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    return model_metrics, best_ransac