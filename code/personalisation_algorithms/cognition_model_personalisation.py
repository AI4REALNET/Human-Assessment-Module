import os
import joblib
from code.utils.ml.quantification.bayesian_regression import bayesian_regression
from code.utils.ml.quantification.linear_regression import linear_regression
from code.utils.ml.quantification.polynomial_regression import polynomial_regression
from code.utils.ml.quantification.KNR import KNRegression
from code.utils.ml.quantification.SVR import SVRegression
from code.utils.ml.quantification.ridge_regression import ridge_regression
from code.utils.ml.quantification.huber_regression import huber_regression
from code.utils.ml.quantification.RANSAC_regression import RANSAC_regression
from code.utils.dataprocessing.dataset_reading_division import cognition_dataset_reading_division
"""
This script loads a features' dataset, applies multiple quantification algorithms,
evaluates them on the same train/test split, and stores the best model based on RMSE for cognition performance prediction.

Steps:
1. Load and preprocess the dataset (80/20 split).
2. Train and evaluate multiple regression models.
3. Compare models using RMSE.
4. Save the best model as a .pkl file with user-defined identifier.
"""
# Step 1: Load dataset and split
X_train, X_test, y_train, y_test = cognition_dataset_reading_division()

# List of regression functions
functions = [
    bayesian_regression,
    linear_regression,
    polynomial_regression,
    KNRegression,
    SVRegression,
    ridge_regression,
    huber_regression,
    RANSAC_regression
]

# Step 2: Train and evaluate all models
eval_results = []
for func in functions:
    print(f"\n{func.__name__}")  
    print("-------------------")
    
    eval_result, best_model, features = func(X_train, X_test, y_train, y_test)  # Call model function

    # Store evaluation results for later comparison
    eval_results.append({
        'Model': func.__name__, 
        'RMSE': eval_result.get('RMSE'), 
        'model_obj': best_model
        })

    # Print evaluation metrics
    print(f"R2: {eval_result.get('R2')}")   
    print(f"RMSE: {eval_result.get('RMSE')}")

# Step 3: Sort models by RMSE (ascending) and select the best one
eval_results_sorted = sorted(eval_results, key=lambda x: x['RMSE'])

print("-------------------")
print("-------------------")
print('Best Model')
print(f"Model: {eval_results_sorted[0].get("Model")}")
print(f"RMSE: {eval_results_sorted[0].get("RMSE")}")

# Get best model and features
best_model_obj = eval_results_sorted[0].get("model_obj")
best_model_features = eval_results_sorted[0].get("Features")

# Step 4: Save model locally with unique controller ID
save_path = "cognition_personalised_models/"
while True:
    # Ask user for a unique controller code
    input_controller_code = input("Insert a numeric code to identify the controller:\n")
    pkl_file_name = os.path.join(save_path, f'model_C{input_controller_code}.pkl')

    if not os.path.exists(pkl_file_name):
        break # Unique filename confirmed
    else:
        print(f"File model_C{input_controller_code}.pkl already exists. Please choose another code.\n")

joblib.dump((best_model_obj, best_model_features), pkl_file_name)
print(f"Model saved to: {pkl_file_name}")