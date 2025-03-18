import numpy as np
from sklearn.metrics import mean_squared_error

def nrmse(y_true: list, y_pred: list, normalization:str = 'std'):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) using sklearn.
    
    :param list y_true: array-like, true values
    :param list y_pred: array-like, predicted values
    :param str normalization: string, 'std' for normalization by standard deviation or 'range' for normalization by range of y_true
    
    :return float NRMSE: float, Normalized RMSE value
    """
    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Normalize the RMSE
    if normalization == 'std':
        # Normalization by standard deviation
        scale = np.std(y_true)
    elif normalization == 'range':
        # Normalization by range (max - min)
        scale = np.max(y_true) - np.min(y_true)
    else:
        raise ValueError("Normalization must be 'std' or 'range'.")
    
    # Calculate NRMSE
    nrmse = rmse / scale
    return nrmse