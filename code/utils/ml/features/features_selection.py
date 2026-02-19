import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE    
from sklearn.feature_selection import mutual_info_regression

def mutual_info_features_selection(X: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    """
    Selects features based on Mutual Information scores.
    
    :param pd.DataFrame X: Feature dataset.
    :param pd.DataFrame Y: Label variable.
    
    :return dict[str, float] mutual_info_scores: Dictionary of features and their mutual information scores
    """
    # Feature Evaluation using Mutual Information
    mutual_info = mutual_info_regression(X, Y)

    # Create a pandas Series with scores and sort descending
    mutual_info_scores = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

    return mutual_info_scores.to_dict()


def select_n_features(test: str, dictionary: dict[str, float], n_features_to_select: int) -> list[tuple[str, float]]:
    """
    Selects the top n_features_to_select features based on the specified statistical test.
    
    :param str test: The type of statistical test used ('correlation', 'mutual_info', 'anova_kuskis', or 'spearman')
    :param dict dictionary[str, float]: Dictionary mapping feature names to importance scores.
    :param int n_features_to_select: Number of top features to select.
    
    :return list top_n_features: List of top N selected features with weights.
    """
    if test == 'correlation' or test == 'spearman' or 'anova_kurkis':
         # Sort the dictionary by values in ascending order
        sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1]))

        # Select the top n_features keys
        top_n_features = list(sorted_dict.keys())[:n_features_to_select]

        return top_n_features
    elif test == 'mutual_info': # mutual info case
        # Sort the dictionary by values in descending order
        sorted_dict = list(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

        return sorted_dict[:n_features_to_select]
    else: 
        raise ValueError("Test type must be 'correlation', 'mutual_info', 'anova_kuskis' or 'spearman'.")

def log_reg_features_selection(X: pd.DataFrame, Y: pd.DataFrame, n_features_to_select: int) -> list[tuple[str, float]]:
    """
    Select top features using Recursive Feature Elimination (RFE) with logistic regression.

    This method fits a logistic regression model and recursively removes the least important features.

    :param X: Feature dataset.
    :param Y: Target variable (must be categorical or binary).
    :param n_features_to_select: Number of features to select.
    
    :return list selected_features: List of selected feature names and regression coefficient as weights.
    """
    # Define logistic regression model
    log_reg = LogisticRegression(solver='liblinear')

    # Use RFE (Recursive Feature Elimination) to rank and select top features
    rfe = RFE(log_reg, n_features_to_select=n_features_to_select)
    rfe.fit(X, Y)

    # Extract selected feature names and weights
    selected_features = X.columns[rfe.support_].tolist()
    weights = abs(rfe.estimator_.coef_[0])
    features_and_weights = list(zip(selected_features, weights))

    return features_and_weights