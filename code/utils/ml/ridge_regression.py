import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

def ridge_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object]:
    """
    Training of Ridge Regression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    A linear regression method with L2 regularization to prevent overfitting.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - Ridge (object): Trained Ridge regression model from `sklearn.linear_model`.
    """

    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)        

    # Define Ridge regression model
    ridge = Ridge()

    # Define parameter grid for alpha tuning
    param_grid = {'alpha': np.logspace(-3, 3, 10)}

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        ridge, 
        param_grid, 
        cv=2, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_ridge = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")

    # Make predictions
    y_pred = best_ridge.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    return model_metrics, best_ridge