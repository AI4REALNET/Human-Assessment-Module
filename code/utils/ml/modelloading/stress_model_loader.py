from pathlib import Path
import pandas as pd
import joblib
from code.utils.dataprocessing.data_parser import get_controller_detailed_info
from code.utils.ml.explainability.models_explainability import stress_explainability

def load_stress_models(controller_id: str, df_features: pd.Dataframe) -> tuple[int, str]:
    """
    Load all personalized stress classification models for a given controller and 
    generate a final ensemble prediction with explainability.

    :param str controller_id: Unique identifier used to load the corresponding model directory.
    :param df_features: DataFrame containing input features for prediction.
    
    :return: 
        - stress_prediction (int): Final ensemble prediction (0 or 1).
        - stress_explainability_str (str): Explanation of the prediction with confidence.
    """
    # Define path to the root of the project
    code_root = Path(__file__).resolve().parents[3]

    # Get all saved model files (.pkl) and the associated CSV weight file
    folder_name = f"Controller_{controller_id}"
    controller_path = code_root / "stress_personalised_models" / folder_name
    stress_pkl_models = list(controller_path.glob("*.pkl"))
    controller_details_file = list(controller_path.glob("*.csv"))
    
    # Load model weight information from CSV
    models_details = get_controller_detailed_info(controller_details_file[0])

    # Loop through models and fit inputed features
    for stress_model in stress_pkl_models:
        # Extract the model name between stress_ and before _C
        stress_model = str(stress_model)
        model_name_start = stress_model.rfind('stress_') + len('stress_')
        model_name_end = stress_model.find('_C', model_name_start)
        model_name = stress_model[model_name_start:model_name_end]

        # Load model and its selected features
        model, features = joblib.load(stress_model)
        X_input = df_features[features]

        # Predic stress
        y_pred = model.predict(X_input)

        models_details[model_name]['Prediction'] = int(y_pred)
    
    # Save prediction result in the models_details dictionary
    stress_prediction, stress_explainability_str = stress_explainability(models_details)

    return stress_prediction, stress_explainability_str