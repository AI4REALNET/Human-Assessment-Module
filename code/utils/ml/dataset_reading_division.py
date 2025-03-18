import pandas as pd
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from code.utils.preprocessing.data_parser import read_datasets
from code.utils.ml.features_selection import select_n_features, mutual_info_features_selection

def dataset_reading_division() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Load, preprocess, and split a dataset into training and test sets.

     The function allows the user to select a JSON file containing the dataset.
     It extracts 5 features, using a statistical method - mutual information - to avoid overfitting to model.
     Then the dataset is splited in 80% training + 20% test.

    :return: A tuple containing:
        - X_train (pd.DataFrame): Training feature set.
        - X_test (pd.DataFrame): Testing feature set.
        - y_train (pd.DataFrame): Training labels.
        - y_test (pd.DataFrame): Testing labels. 
    """
    # Open a file dialog to select JSON dataset file
    df = filedialog.askopenfilename(title="Select dataset json files", filetypes=[("JSON", ".json")])
    
    # Open JSON features file 
    df = read_datasets(df)

    # Filter only CRTTs data
    dataset = df[df['Test_phase'].isin(['CRTT1', 'CRTT2'])]

    # Cognitive performance - label
    y = dataset['Cognitive_performance']

    # Drop all other columns other than features
    X = dataset.drop(columns=['Test_phase', 'STAI_6items', 'ECG', 'VAS', 'Accuracy', 'Reaction_time', 'RT_std', 'Cognitive_performance'])

    # Feature selection
    features = select_n_features('mutual_info', mutual_info_features_selection(X, y), 5)

    print('Selected Features:')
    print(features)

    # Split the dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test