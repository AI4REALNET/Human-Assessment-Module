import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from scipy.stats import f_oneway, kruskal, shapiro

def features_correlation(X: pd.DataFrame, threshold: float) -> dict[str, float]:
    """
    Identifies features with correlation below a given threshold.
    
    :param pd.DataFrame X: Feature dataset
    :param float threshold: Correlation threshold value
    
    :return dict[str, float] selected_features: Dictionary with selected features and their max correlation value
    """
    # Correlation between X features
    corr_matrix = X.corr()

    # Exclude the diagonal by setting it to 0
    corr_matrix.values[np.triu_indices_from(corr_matrix)] = 0

    # Dictionary to store features and their maximum correlation value
    selected_features = {}

    for column in corr_matrix.columns:
        # Find the maximum absolute correlation value for each feature
        max_corr = abs(corr_matrix[column]).max()
        
        # Add to the dictionary if it is below the threshold
        if max_corr < threshold:
            selected_features[column] = max_corr

    return selected_features

def mutual_info_features_selection(X: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    """
    Selects features based on Mutual Information scores.
    
    :param pd.DataFrame X: Feature dataset
    :param pd.DataFrame Y: Label variable
    
    :return dict[str, float] mutual_info_scores: Dictionary of features and their mutual information scores
    """
    # Feature Evaluation using Mutual Information
    mutual_info = mutual_info_regression(X, Y)
    mutual_info_scores = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

    return mutual_info_scores.to_dict()

def spearman_feature_selection(X: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    """
    Selects features using Spearman correlation with a significance threshold of 0.05.
    
    :param pd.DataFrame X: Feature dataset
    :param pd.DataFrame Y: Target variable
    
    :return dict[str, float] selected_features: Dictionary with selected features and their p-values
    """
    selected_features = {}

    # Check Spearman correlation for each feature
    for feature in X.columns:
        _, p_value = spearmanr(X[feature], Y)

        # Select feature if p-value < 0.05
        if p_value < 0.05:
            selected_features[feature] = p_value

    return selected_features

def feature_class_differentiation(X: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    """
    Evaluates feature differentiation across classes using ANOVA or Kruskal-Wallis , acording to features distribution.
    
    :param pd.DataFrame X: Feature dataset
    :param pd.DataFrame Y: Target categorical variable
    
    :return dict[str, float] selected_features: Dictionary with selected features and their p-values
    """
    selected_features = {}

    # Ensure Y is a pandas Series to use .unique()
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y)
        
    # Check each feature
    for feature in X.columns:
        feature_data = X[feature]

        # Test for normality using Shapiro-Wilk test
        _, p_normality = shapiro(feature_data)

        if p_normality > 0.05:
            # Feature is normally distributed, apply ANOVA
            _, p_value = f_oneway(*[feature_data[Y == cls] for cls in Y.unique()])

            if p_value < 0.05:
                selected_features[feature] = p_value
        else:
            # Feature is not normally distributed, apply Kruskal-Wallis
            _, p_value = kruskal(*[feature_data[Y == cls] for cls in Y.unique()])

            if p_value < 0.05:
                selected_features[feature] = p_value

    return selected_features

def select_n_features(test: str, dictionary: dict[str, float], n_features_to_select: int) -> list:
    """
    Selects the top n_features_to_select features based on the specified statistical test.
    
    :param str test: The type of statistical test used ('correlation', 'mutual_info', 'anova_kuskis', or 'spearman')
    :param dict dictionary[str, float]: Dictionary of feature scores
    :param int n_features_to_select: Number of top features to select
    
    :return list top_n_features: List of top N selected features
    """
    if test == 'correlation' or test == 'spearman' or 'anova_kurkis':
         # Sort the dictionary by values in ascending order
        sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1]))

        # Select the top n_features keys
        top_n_features = list(sorted_dict.keys())[:n_features_to_select]

        return top_n_features
    elif 'mutual_info': # mutual info case
        # Sort the dictionary by values in descending order
        sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

        # Select the top n_features keys
        top_n_features = list(sorted_dict.keys())[:n_features_to_select]
        return top_n_features
    else: 
        raise ValueError("Test type must be 'correlation', 'mutual_info', 'anova_kuskis' or 'spearman'.")