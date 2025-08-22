import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import root_mean_squared_error, r2_score

def polynomial_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Training of Polynomial Regression with hyperparameters tuning 
    and evaluation of performance in Test set.
    
    A regression technique that models relationships using polynomial functions of the independent variable.
    
    :param pd.DataFrame X_train: Training feature set
    :param pd.DataFrame X_test: Testing feature set
    :param pd.DataFrame y_train: Training label (CP) values
    :param pd.DataFrame y_test: Testing label (CP) values

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary containing model evaluation metrics:
            - "RMSE" (float): Root Mean Squared Error.
            - "R2" (float): R-squared score.
        - Pipeline (object): Trained Polynomial model from `sklearn.preprocessing` and `sklearn.linear_model`.
        - features (list[str]): List of feature names used during testing.
    """
    # Define polynomial features transformation
    poly = PolynomialFeatures()

    # Scale the features using StandardScaler after generating polynomial features
    scaler = StandardScaler()

    # Define hyperparameter grid including degree of polynomial
    param_grid = {
        'polynomialfeatures__degree': [2, 3, 4],  # Test different degrees of polynomial features
        'linearregression__fit_intercept': [True, False],  # Whether to include intercept
    }

    # Create a pipeline for transforming features and applying LinearRegression
    pipeline = Pipeline([
        ('polynomialfeatures', poly),
        ('scaler', scaler),
        ('linearregression', LinearRegression())
    ])

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=2,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    model_metrics = {}
    model_metrics['RMSE'] = root_mean_squared_error(y_test, y_pred)
    model_metrics['R2'] = r2_score(y_test, y_pred)

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_model, features