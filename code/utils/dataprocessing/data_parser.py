import pandas as pd

def read_datasets(json_file_path: str) -> pd.DataFrame:
    """ Read dataset from a JSON file into a pandas DataFrame.

    This function reads a JSON file from the specified path and loads it 
    into a pandas DataFrame using the 'index' orientation for proper indexing.

    :param str json_file_path: Path to the JSON file
    :return pd.Dataframe df: DataFrame containing the data from the JSON file
    """
    # Read JSON using 'index' orientation — assumes keys are row indices
    df = pd.read_json(json_file_path, orient="index")

    return df

def get_controller_detailed_info(csv_path: str) -> dict[str, dict[str, float]]: 
    """
    Retrieve models weight details from CSV file.

    Reads a CSV file where each row contains the name of a model and its corresponding weight,
    and builds a dictionary mapping model names to their weights.
    
    :param str csv_path: PAth to the CSV file.

    :return dict[str, dict[str, float]]: A dictionary with model names as keys and
        a nested dictionary containing the weight under the 'weight' key.
        Example: {'ModelA': {'weight': 0.75}, 'ModelB': {'weight': 0.25}} 
    """
    models_details = {}
    # Open the CSV file and read contents using pandas
    with open(csv_path, mode='r') as file:
        csv_reader = pd.read_csv(file)

        # Iterate through each row to extract model name and weight
        for _, row in csv_reader.iterrows():
            model_name = row['Model']  # Gets 'Model' column in CSV
            weight = row['Weight'] # Gets 'Weight' column in CSV

            # Store weight inside a nested dictionary
            models_details[model_name] = {'weight': weight}

    return models_details