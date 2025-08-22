import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

def bayesian_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Training of Bayesian Regression with hyperparameters tuning 
    and evaluation of performance in Test set.

    A probabilistic approach to regression that updates beliefs using Bayes' theorem.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - BayesianRidge (object): Trained Bayesian Ridge regression model from `sklearn.linear_model`.
        - features (list[str]): List of feature names used during testing.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Bayesian Ridge Regression model
    bayesian_ridge = BayesianRidge()

    # Define parameter grid for tuning
    param_grid = {
        'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],  # Hyperparameter controlling alpha prior
        'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],  # Hyperparameter controlling alpha variance
        'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],  # Hyperparameter controlling lambda prior
        'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3]   # Hyperparameter controlling lambda variance
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        bayesian_ridge, 
        param_grid, 
        cv=2, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_bayesian = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions
    y_pred = best_bayesian.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_bayesian, features