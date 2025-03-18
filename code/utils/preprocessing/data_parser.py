import pandas as pd

def read_datasets(json_file_path: str) -> pd.DataFrame:
    """ Read dataset from a JSON file into a pandas DataFrame.

    This function reads a JSON file from the specified path and loads it 
    into a pandas DataFrame using the 'index' orientation for proper indexing.

    :param str json_file_path: Path to the JSON file
    :return pd.Dataframe df: DataFrame containing the data from the JSON file
    """
    df = pd.read_json(json_file_path, orient="index")

    return df