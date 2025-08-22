import pandas as pd
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from code.utils.dataprocessing.data_parser import read_datasets
from code.utils.ml.features.features_selection import select_n_features, mutual_info_features_selection, log_reg_features_selection

def cognition_dataset_reading_division() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Load, preprocess, and split a dataset into training and test sets.

    The function allows the user to select a JSON dataset file.
    It extracts 5 features, using a statistical method - mutual information - to avoid overfitting to model.
    Then the dataset is splited in 80% training + 20% test.

    :return: A tuple containing:
        - X_train (pd.DataFrame): Training feature set.
        - X_test (pd.DataFrame): Testing feature set.
        - y_train (pd.DataFrame): Training labels.
        - y_test (pd.DataFrame): Testing labels. 
    """
    # Open a file dialog to select a JSON dataset file
    json_file = filedialog.askopenfilename(title="Select dataset json file", filetypes=[("JSON", ".json")])

    # Read the dataset and parse it into a DataFrame
    df = read_datasets(json_file)

    # Filter only CRTTs data
    dataset = df[df['Test_phase'].isin(['CRTT1', 'CRTT2'])]

    # Define labels - Cognitive performance
    y = dataset['Cognitive_performance']

    # Drop all other columns not used for modeling
    X = dataset.drop(columns=['Test_phase', 'STAI_6items', 'ECG', 'VAS', 'Accuracy', 'Reaction_time', 'RT_std', 'Cognitive_performance'])

    # Select top 5 features using mutual information criterion
    features = select_n_features('mutual_info', mutual_info_features_selection(X, y), 5)

    # Split the dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def stress_dataset_reading_division() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Load, preprocess, and split a dataset into training and test sets.

    The function allows the user to select a JSON file containing the dataset.
    It extracts 5 features, using a logistic regression method for feature selection.
    Then the dataset is splited in 80% training + 20% test.

    :return: A tuple containing:
        - X_train (pd.DataFrame): Training feature set.
        - X_test (pd.DataFrame): Testing feature set.
        - y_train (pd.DataFrame): Training labels.
        - y_test (pd.DataFrame): Testing labels. 
    """
    # Open a file dialog to select a JSON dataset file
    json_file = filedialog.askopenfilename(title="Select dataset json file", filetypes=[("JSON", ".json")])

    # Read the dataset and parse it into a DataFrame
    df = read_datasets(json_file)

    # Extract dataset identifier from the filename
    file_name = json_file.rsplit('/', 1)[-1]
    ATC_n = file_name.split('_')[1]
    print("-" * 30)
    print(f"ATC {ATC_n}")

    # Define labels - stress 
    y = df['Test_phase']

    # Encode target labels `y` using Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Drop all other columns not used for modeling
    X = df.drop(columns=['Test_phase', 'STAI_6items', 'ECG', 'VAS', 'Accuracy', 'Reaction_time', 'RT_std', 'Cognitive_performance'])

    # Select top 5 features using logistic regression-based selection
    features = log_reg_features_selection(X, y_encoded, 5)

    # Split the dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X[features], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    return X_train, X_test, y_train, y_test