import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def decision_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Train a Decision Tree classifier with hyperparameter tuning 
    and evaluate its performance on the test set.

    Decision Tree is a regression or classification method that splits data into branches based on feature values,
    creating a tree-like structure to make decisions.

    :param X_train: Training feature set.
    :param X_test: Testing feature set.
    :param y_train: Training labels.
    :param y_test: Testing labels.

    :return: A tuple containing:
        - model_metrics (dict[str, float]): Dictionary of performance metrics:
            - "Balanced Accuracy"
            - "Weighted Precision"
            - "Weighted Recall"
            - "Weighted F1-Score"
        - best_dt_classifier (object): Best fitted DecisionTreeClassifier model.
        - features (list[str]): List of feature names used in the model.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Decision Tree model
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [5, 10, 15, None],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        dt_classifier,
        param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_dt_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions on test set
    y_pred = best_dt_classifier.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['Balanced Accuracy'] = balanced_accuracy_score(y_test, y_pred)
    model_metrics['Weighted Precision'] = precision_score(y_test, y_pred, average='weighted') 
    model_metrics['Weighted Recall'] = recall_score(y_test, y_pred, average='weighted')
    model_metrics['Weighted F1-Score'] = f1_score(y_test, y_pred, average='weighted')

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_dt_classifier, features