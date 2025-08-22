import time
import pandas as pd
from code.utils.ml.modelloading.stress_model_loader import load_stress_models
from code.utils.ml.modelloading.cognition_model_loader import load_cognition_model

controller_id = input("Inser the numerical code of the controller to fetch their personalised model:\n")

while True:
    start_time = time.time() # Start clock for each loop 

    # TODO: Replace with automated new features loading every 1min
    df_features = pd.read_csv("example_ecg_features.csv")

    # Run cognition and stress models prediction
    cog_pred, cog_explainability = load_cognition_model(controller_id, df_features)
    stress_pred, stress_explainability = load_stress_models(controller_id, df_features)

    # Print Cognition Output
    print(cog_explainability)
    # Print Stress Output
    print(stress_explainability)
    
    # Wait to complete a full 60-second cycle (processing time included)
    elapsed = time.time() - start_time
    time_to_wait = max(0, 60 - elapsed)
    time.sleep(time_to_wait)