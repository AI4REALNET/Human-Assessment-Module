import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

def linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Training of Linear Regression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    A regression method that models the relationship between variables using a linear equation.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - LinearRegression (object): Trained LinearRegression model from `sklearn.linear_model`.
        - features (list[str]): List of feature names used during testing.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_reg = LinearRegression()

    # Define hyperparameter grid
    param_grid = {
        'fit_intercept': [True, False],  # Whether to include intercept
    }

    # Perform grid search
    grid_search = GridSearchCV(
        lin_reg,
        param_grid=param_grid,
        scoring='r2',
        cv=2,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model on the training data
    grid_search.fit(X_train_scaled, y_train)

    # Retrieve the best estimator
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_model, features