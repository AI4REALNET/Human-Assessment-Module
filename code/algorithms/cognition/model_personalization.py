from code.utils.ml.dataset_reading_division import dataset_reading_division
from code.utils.ml.bayesian_regression import bayesian_regression
from code.utils.ml.linear_regression import linear_regression
from code.utils.ml.polynomial_regression import polynomial_regression
from code.utils.ml.KNR import KNRegression
from code.utils.ml.SVR import SVRegression
from code.utils.ml.ridge_regression import ridge_regression
from code.utils.ml.huber_regression import huber_regression
from code.utils.ml.RANSAC_regression import RANSAC_regression
import joblib
import os

# Load dataset and split into training and testing sets
X_train, X_test, y_train, y_test = dataset_reading_division()

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

# Iterate over functions and print results
eval_results = []
for func in functions:
    print(f"\n{func.__name__}")  
    print("-------------------")
    
    eval_result, best_model = func(X_train, X_test, y_train, y_test)  # Call function
    eval_results.append({'Model': func.__name__, 'RMSE': eval_result.get('RMSE'), 'model_obj': best_model})

    print(f"R2: {eval_result.get('R2')}")   
    print(f"RMSE: {eval_result.get('RMSE')}")

# Sort eval_results by RMSE (ascending order) and show best one
eval_results_sorted = sorted(eval_results, key=lambda x: x['RMSE'])

print("-------------------")
print("-------------------")
print('Best Model')
print(f"Model: {eval_results_sorted[0].get("Model")}")
print(f"RMSE: {eval_results_sorted[0].get("RMSE")}")


best_model_obj = eval_results_sorted[0].get("model_obj")

# Dsefine folder path ans unique code for the model
save_path = "personalised_models"
while True:
    input_controller_code = input("Insert a numeric code to identify the controller:\n")
    pkl_file_name = os.path.join(save_path, f'model_C{input_controller_code}.pkl')

    if not os.path.exists(pkl_file_name):
        print(f"File 'model_C{input_controller_code}.pkl' save with success in {pkl_file_name}!")
        break
    else:
        print(f"File model_C{input_controller_code}.pkl already exists. Please choose another code.\n")

# Save the best model using joblib
joblib.dump(best_model_obj, pkl_file_name)