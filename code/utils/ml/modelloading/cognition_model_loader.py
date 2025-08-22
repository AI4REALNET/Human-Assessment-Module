from pathlib import Path
import pandas as pd
import joblib
from code.utils.ml.explainability.models_explainability import cognition_explainability

def load_cognition_model(controller_id: str, df_features: pd.Dataframe) -> tuple[float, str]:
    """
    Load a personalised cognition model for a given controller and predict cognitive performance.

    :param str controller_id: Unique identifier used to load the corresponding model file.
    :param pd.Dataframe df_features: DataFrame containing all available features for prediction.
    
    :return:
        - cognition_prediction (float): Predicted cognition performance value (bounded between 0–1).
        - cognition_explainability_str (str): Explainability message for the prediction.
    """
    # Define path to the root of the project
    code_root = Path(__file__).resolve().parents[3]

    # Construct the path to the saved model using the controller ID
    filename = f"cognition_model_C{controller_id}.pkl"
    pkl_model_path = code_root / "cognition_personalised_models" / filename

    # Load model and its feature list
    model, features = joblib.load(pkl_model_path)

    # Make prediction with input signal
    X_input = df_features[features]
    y_pred = model.predict(X_input)

    # Convert model prediction into a cognition index + explainability
    cognition_prediction, cognition_explainability_str = cognition_explainability(y_pred)

    return cognition_prediction, cognition_explainability_str