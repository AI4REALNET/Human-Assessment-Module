import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def SVM(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[dict[str, float], object, list[str]]:
    """
    Train a Support Vector Machine (SVM) classifier with hyperparameter tuning 
    and evaluate its performance on the test set.

    SVM is a classification algorithm that finds the hyperplane which best separates 
    classes in the feature space, using different kernel functions for flexibility.

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
        - best_svm (object): Trained SVC model with best hyperparameters.
        - features (list[str]): List of feature names used in the model.
    """
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define SVM model
    svm = SVC(decision_function_shape='ovr', random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
        'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    }

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with the training data
    grid_search.fit(X_train_scaled, y_train)  # Flatten y_train for compatibility

    # Get the best model and its parameters
    best_svm = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Make predictions on test set
    y_pred = best_svm.predict(X_test_scaled)

    # Evaluate the model
    model_metrics = {}
    model_metrics['Balanced Accuracy'] = balanced_accuracy_score(y_test, y_pred)
    model_metrics['Weighted Precision'] = precision_score(y_test, y_pred, average='weighted') 
    model_metrics['Weighted Recall'] = recall_score(y_test, y_pred, average='weighted')
    model_metrics['Weighted F1-Score'] = f1_score(y_test, y_pred, average='weighted')

    # List with used features
    features = X_test.columns.tolist()

    return model_metrics, best_svm, features