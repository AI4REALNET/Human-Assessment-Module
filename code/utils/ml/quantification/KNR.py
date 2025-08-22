import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from math import floor

def KNRegression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Training of KN Ridge Regression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    A non-parametric regression using k-nearest neighbors for prediction.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - KNeighborsRegressor (object): Trained KNeighborsRegressor model from `sklearn.neighbors`.
        - features (list[str]): List of feature names used during testing.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the KNN regressor model
    knn = KNeighborsRegressor()

    # Define parameter grid for hyperparameter tuning
    cross_validation_folds = 2
    #Define neibours and avoid error of range(2,2)
    neibours = range(2, floor(len(X_train_scaled) / cross_validation_folds))
    if neibours == range(2,2):
        neibours = range(1,2)
    param_grid = {
        'n_neighbors': neibours,
        'weights': ['uniform', 'distance'],  # Weighting method
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
        'leaf_size': range(20, 100, 5),  # Leaf size for the tree used by KNN
        'p': [1, 2]  # Distance metric (1: Manhattan, 2: Euclidean)
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        knn, 
        param_grid, 
        cv=cross_validation_folds, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_knn = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions
    y_pred = best_knn.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_knn, features