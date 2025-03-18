import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

def SVRegression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object]:
    """
    Training of SVRegression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    Support Vector Regression (SVR) is a regression method that uses Support Vector Machines to find a hyperplane minimizing error within a margin.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - SVR (object): Trained SVR regression model from `sklearn.svm`.
    """

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the SVR model
    svr = SVR()

    # Define parameter grid for SVR hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'epsilon': [0.01, 0.1, 0.2, 0.5],  # Defines the margin of tolerance
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Different kernel functions
        'degree': [2, 3, 4], # Degree for polynomial kernel (if used)
        'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' and 'poly'
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        svr, 
        param_grid, 
        cv=2, 
        scoring='neg_root_mean_squared_error',   
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_svr = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions
    y_pred = best_svr.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    return model_metrics, best_svr