import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def gradient_boosting(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Train a Gradient Boosting classifier with hyperparameter tuning 
    and evaluate its performance on the test set.

    Gradient Boosting is an ensemble method that builds a series of decision trees, 
    where each tree corrects the errors of the previous one to improve accuracy.

    :param X_train: Training feature set.
    :param X_test: Testing feature set.
    :param y_train: Training labels.
    :param y_test: Testing labels.

    :return: A tuple containing:
        - model_metrics (dict): Dictionary of performance metrics:
            - "Balanced Accuracy"
            - "Weighted Precision"
            - "Weighted Recall"
            - "Weighted F1-Score"
        - best_gb_classifier (object): Trained GradientBoostingClassifier with best hyperparameters.
        - features (list[str]): List of feature names used in the model.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Gradient Boosting model
    gb_classifier = GradientBoostingClassifier(random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages
        'learning_rate': [0.01, 0.1, 0.2],  # Shrinks the contribution of each tree
        'max_depth': [3, 5, 10],  # Maximum depth of individual trees
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples at a leaf node
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        gb_classifier,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_gb_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions on test set
    y_pred = best_gb_classifier.predict(X_test_scaled)
    
    # Evaluate the model
    model_metrics = {}
    model_metrics['Balanced Accuracy'] = balanced_accuracy_score(y_test, y_pred)
    model_metrics['Weighted Precision'] = precision_score(y_test, y_pred, average='weighted') 
    model_metrics['Weighted Recall'] = recall_score(y_test, y_pred, average='weighted')
    model_metrics['Weighted F1-Score'] = f1_score(y_test, y_pred, average='weighted')

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_gb_classifier, features