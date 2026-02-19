import os
import joblib
import pandas as pd
import openpyxl
from code.utils.ml.classification.decision_tree import decision_tree
from code.utils.ml.classification.gradient_boosting import gradient_boosting
from code.utils.ml.classification.KNN import KNN
from code.utils.ml.classification.LDA import LDA
from code.utils.ml.classification.logistic_regression import logistic_regression
from code.utils.ml.classification.naive_bayes import naive_bayes
from code.utils.ml.classification.SVM import SVM
from code.utils.dataprocessing.dataset_reading_division import stress_dataset_reading_division
"""
This script loads a features' dataset, applies multiple classification models,
evaluates them based on weighted F1-score and balanced accuracy, stores each stress model 
along with its features, and creates a CSV with the relative performance weights 
for explainability or ensemble usage.

Steps:
1. Load and split dataset (80/20).
2. Train and evaluate each classifier.
3. Save each model + selected features.
4. Compute F1-score-based weights for each model.
5. Compute overall evaluation metrics.
6. Save weights as a CSV file.
"""
# Step 1: Load stress dataset
X_train, X_test, y_train, y_test, features_and_weights = stress_dataset_reading_division()

# List of classifiers
functions = [
    logistic_regression,
    KNN,
    SVM,
    decision_tree,
    gradient_boosting,
    naive_bayes,
    LDA,
]

# Ask user for a unique controller code
input_controller_code = input("Insert a numeric code to identify the controller:\n")
save_path = "stress_personalised_models/" # saving path

# Create folder to save controller's files all together
folder_dir = os.path.join(save_path, 'Controller_' + str(input_controller_code))
os.makedirs(folder_dir, exist_ok=True)

# Step 2: Train and evaluate all classifiers
eval_results = []
for func in functions: # Iterate over each function (classifier) to evaluate performance
    print(f"\n{func.__name__}")  
    print("-------------------")
    
    eval_result, best_model, features = func(X_train, X_test, y_train, y_test)  # Call model function
    
    # Store the results (model name, weighted F1-score, and model object)
    eval_results.append({
        'Model': func.__name__, 
        'Weighted F1-Score': eval_result.get('Weighted F1-Score'), 
        'Balanced Accuracy': eval_result.get('Balanced Accuracy'),
        'Weighted TPR': eval_result.get('Weighted Recall'),
        'model_obj': best_model, 
        'Features': features
        })

    # Print evaluation metrics
    print(f"Balanced Accuracy: {eval_result.get('Balanced Accuracy')}")   
    print(f"Weighted F1-Score: {eval_result.get('Weighted F1-Score')}")
    print(f"Weighted TPR: {eval_result.get('Weighted Recall')}")

    # Step 3: Save model + selected features as a .pkl file
    pkl_file_name = os.path.join(folder_dir, 'stress_' + func.__name__ + '_C' + str(input_controller_code)  + '.pkl')
    joblib.dump((best_model, features), pkl_file_name)

# Step 4: Compute F1-score-based weights for explainability
# Create a dictionary mapping models to F1-scores
dict_models_f1score = {result['Model']: result['Weighted F1-Score'] for result in eval_results}

# List of Weighted F1-Scores
f1_scores_list = [result['Weighted F1-Score'] for result in eval_results]

# Calculate the total sum of the Weighted F1-Scores (for normalization)
total_f1_scores = sum(f1_scores_list)

# Calculate the weight for each model as a proportion of its F1 score to the total sum
models_weights = [(score / total_f1_scores) for score in f1_scores_list]
rounded_weights = [round(weight, 2) for weight in models_weights] # Round the weights to 2 decimal places

model_weights_rounded = [{
    'Model': eval_results[i]['Model'],
    'Weight': rounded_weights[i]
} for i in range(len(eval_results))]
df_weights = pd.DataFrame(model_weights_rounded)

# Step 5: Computing average evaluation metrics for all models
TPR_list = [result['Weighted TPR'] for result in eval_results]
avg_TPR = sum(TPR_list)/float(len(TPR_list))

ACC_list = [result['Balanced Accuracy'] for result in eval_results]
avg_ACC = sum(ACC_list)/float(len(ACC_list))

avg_F1 = sum(f1_scores_list)/float(len(f1_scores_list))

# Step 6: Saving and storing
# Save in .csv file models and explainability weights
csv_name = os.path.join(folder_dir, 'model_weights__C' + str(input_controller_code)  + '.csv')
df_weights.to_csv(csv_name, index=False) # Save to CSV without row indices

# Save in .xlsx file models properties and performance
modelinfo_file_name = os.path.join(save_path, 'modelinfo_C' + str(input_controller_code) + '.xlsx')
modelinfo = pd.DataFrame([avg_TPR, avg_ACC, avg_F1])
featselres = pd.DataFrame(features_and_weights)

with pd.ExcelWriter(modelinfo_file_name) as writer:
    modelinfo.to_excel(writer, sheet_name='Model Properties',engine="openpyxl", index=False, header=False)
    featselres.to_excel(writer, sheet_name='Features',engine="openpyxl", index=False, header=False)