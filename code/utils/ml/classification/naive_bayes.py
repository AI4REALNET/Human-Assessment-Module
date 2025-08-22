import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def naive_bayes(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Train a Gaussian Naive Bayes classifier with hyperparameter tuning 
    and evaluate its performance on the test set.

    Naive Bayes is a probabilistic classification method that assumes 
    independence between features and applies Bayes' theorem to make predictions.

    :param X_train: Training feature set.
    :param X_test: Testing feature set.
    :param y_train: Training labels.
    :param y_test: Testing labels.

    :return: A tuple containing:
        - model_metrics (dict): Performance metrics:
            - "Balanced Accuracy"
            - "Weighted Precision"
            - "Weighted Recall"
            - "Weighted F1-Score"
        - best_nb (object): Trained GaussianNB model with best hyperparameters.
        - features (list[str]): List of feature names used in the model.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Naïve Bayes model
    nb = GaussianNB()

    # Define parameter grid for tuning
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Smoothing parameter
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        nb,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with the training data
    grid_search.fit(X_train_scaled, y_train)  # Flatten y_train for compatibility

    # Get the best model and its parameters
    best_nb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions on test set
    y_pred = best_nb.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['Balanced Accuracy'] = balanced_accuracy_score(y_test, y_pred)
    model_metrics['Weighted Precision'] = precision_score(y_test, y_pred, average='weighted') 
    model_metrics['Weighted Recall'] = recall_score(y_test, y_pred, average='weighted')
    model_metrics['Weighted F1-Score'] = f1_score(y_test, y_pred, average='weighted')

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_nb, features