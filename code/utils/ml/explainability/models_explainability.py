def stress_explainability(models_details:dict) -> tuple[int, str]:
    """
    Generate a stress prediction and an explainability statement 
    based on a weighted voting of multiple model predictions.

    Each model provides a binary prediction (0 = no stress, 1 = stress) 
    and a weight (based on performance, e.g. F1-score). The function 
    aggregates the weights to determine the most probable output.

    :param dict models_details: Dictionary containing each model's output and weight.

    :return: 
        - prediction (int): Final prediction (0 or 1).
        - explainability_str (str): Explanation with confidence percentage.
    """
    # Initialize sums for predictions 0 and 1
    sum_prediction_0 = 0
    sum_prediction_1 = 0
    # Iterate over the models in the dictionary
    for _, details in models_details.items():
        if details['Prediction'] == 0:
            sum_prediction_0 += details['weight']
        elif details['Prediction'] == 1:
            sum_prediction_1 += details['weight']

    # Round the sums to 2 decimal places
    sum_prediction_0 = round(sum_prediction_0, 2)
    sum_prediction_1 = round(sum_prediction_1, 2)

    # Verify more probable output and its explainability
    if sum_prediction_0 > sum_prediction_1:
        prediction = 0
        explainability = sum_prediction_0
        if explainability > 0.8:
            explainability_str = "No stress detected. High confidence: it is almost certain there is no stress."
        elif (explainability <= 0.8 and explainability >= 0.65):
            explainability_str = "No stress detected. Moderate confidence: likely no stress, but with some uncertainty."
        else:
            explainability_str = "No stress detected. Low confidence: confidence is limited, stress is close to the threshold."
        
    else:
        prediction = 1
        explainability = sum_prediction_1
        if explainability > 0.8:
            explainability_str = "Stress detected. High confidence: it is strongly certain that stress is present."
        elif (explainability <= 0.8 and explainability >= 0.65):
            explainability_str = "Stress detected. Moderate confidence: stress is likely present, but not definitive."
        else:
            explainability_str = "Stress detected. Low confidence: certainty is limited."

    return prediction, explainability_str

def cognition_explainability(cognition_prediction) -> tuple[float, str]:
    """
    Normalise and explain a cognition performance prediction 
    expressed as a continuous value between 0 and 1.

    If the prediction is out of bounds (<0 or >1), it is clipped to the valid range.

    :param float cognition_prediction: Predicted cognition performance index.

    :return: 
        - cognition_prediction (float): Cognitive performance index between 0 and 1.
        - explainability_str (str): Explanation of index.
    """
    # Make prediction be bounded between 0-1
    if cognition_prediction < 0 : 
        cognition_prediction = 0 
    elif cognition_prediction > 1: 
        cognition_prediction = 1

    # Conditions for explainability
    if cognition_prediction < 0.2:
        explainability_str = "Full support needed: Cognition is impaired. Continuous support is recommended, along with structured tasks."
    elif (cognition_prediction >= 0.2 and cognition_prediction < 0.35):
        explainability_str = "Very limited capacity: Needs constant guidance."
    elif (cognition_prediction >= 0.35 and cognition_prediction < 0.5):
        explainability_str = "Reduced capacity: Benefits from simplification and step-by-step guidance."
    elif (cognition_prediction >= 0.5 and cognition_prediction < 0.65):
        explainability_str = "Partial autonomy: can operate under support, vulnerable under pressure. Regular feedback is beneficial."
    elif (cognition_prediction >= 0.65 and cognition_prediction < 0.8):
        explainability_str = "Stable autonomy: with clear goals and planning can mantain good performance."
    else:
        explainability_str = "High performance: Fast, accurate and can assist others in more demanding tasks."
    return cognition_prediction, explainability_str