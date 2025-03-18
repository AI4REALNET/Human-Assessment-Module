import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

def huber_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object]:
    """
    Training of Huber Regression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    A robust regression method that reduces sensitivity to outliers.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - HuberRegressor (object): Trained HuberRegressor model from `sklearn.linear_model`.
    """

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Huber Regressor
    huber = HuberRegressor()

    # Define parameter grid for tuning
    param_grid = {
        'epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],  # Sensitivity to outliers
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],   # Regularization strength
        'max_iter': [1000, 5000, 10000]  # Increase iterations to ensure convergence
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        huber, 
        param_grid, 
        cv=2, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_huber = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions
    y_pred = best_huber.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    return model_metrics, best_huber