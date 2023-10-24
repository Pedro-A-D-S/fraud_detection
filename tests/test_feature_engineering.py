
import logging
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scripts.FeatureEngineering import FeatureEngineering

logging.basicConfig(
    level=logging.INFO,
    filename='log/FeatureEngineering.log',
    format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')


def test_load_data(fe_instance):
    """
    Test case for the load_data method of the FeatureEngineering class.

    This test verifies whether the load_data method correctly loads the train and test files,
    splits them into features and labels, and returns them as separate variables.
    """

    X_train, X_test, y_train, y_test = fe_instance.load_data()

    # Check if the returned values are not None
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    # Check if the returned valeus are of the correct types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Check if the number of features and labels is correct
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_preprocess_data(fe_instance):
    """
    Test case for the preprocess_data method of the FeatureEngineering class.

    This test verifies whether the preprocess_data method correctly preprocesses the input data
    by applying scaling and encoding transformations.
    """
    # Sample input data
    X_train = pd.DataFrame({
        'tempo': [1, 2, 3],
        'valor': [10, 20, 30],
        'saldo_inicial_c1': [100, 200, 300],
        'novo_saldo_c1': [110, 220, 330],
        'saldo_inicial_c2': [1000, 2000, 3000],
        'novo_saldo_c2': [1100, 2200, 3300],
        'tipo': ['A', 'B', 'C']
    })

    X_test = pd.DataFrame({
        'tempo': [4, 5, 6],
        'valor': [40, 50, 60],
        'saldo_inicial_c1': [400, 500, 600],
        'novo_saldo_c1': [440, 550, 660],
        'saldo_inicial_c2': [4000, 5000, 6000],
        'novo_saldo_c2': [4400, 5500, 6600],
        'tipo': ['D', 'E', 'F']
    })

    nnumeric_features = ['tempo', 'valor', 'saldo_inicial_c1',
                         'novo_saldo_c1', 'saldo_inicial_c2', 'novo_saldo_c2']
    categorical_features = ['tipo']

    X_train_processed, X_test_processed = fe_instance.preprocess_data(
        X_train, X_test, nnumeric_features, categorical_features)

    # Check if the returned valures are not None
    assert X_train_processed is not None
    assert X_test_processed is not None

    # Check if the returned valures are of the correct types
    assert isinstance(X_train_processed, np.ndarray)
    assert isinstance(X_test_processed, np.ndarray)

    # Check if the number of features is the same
    assert X_train_processed.shape[1] == X_test_processed.shape[1]

    # Check if the number of samples is the same
    assert X_train_processed.shape[0] == X_train.shape[0]
    assert X_test_processed.shape[0] == X_test.shape[0]


def test_save_preprocessed_data(fe_instance, tmp_path):
    """
    Test case for the save_preprocessed_data method of the FeatureEngineering class.

    This test verifies whether the save_preprocessed_data method correctly saves the preprocessed data to CSV files.
    """
    # Sample input data
    X_train_processed = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    X_test_processed = np.array([
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ])
    y_train = np.array([0, 1, 0])
    y_test = np.array([1, 0, 1])

    X_train_file = tmp_path / 'X_train.csv'
    X_test_file = tmp_path / 'X_test.csv'
    y_train_file = tmp_path / 'y_train.csv'
    y_test_file = tmp_path / 'y_test.csv'

    fe_instance.save_preprocessed_data(X_train_processed, X_test_processed, y_train, y_test,
                                       str(X_train_file), str(X_test_file), str(y_train_file), str(y_test_file))

    # Check if the CSV files were created
    assert X_train_file.exists()
    assert X_test_file.exists()
    assert y_train_file.exists()
    assert y_test_file.exists()

    # Check if the CSV files have the correct number of rows and columns
    X_train_df = pd.read_csv(X_train_file)
    X_test_df = pd.read_csv(X_test_file)
    y_train_df = pd.read_csv(y_train_file)
    y_test_df = pd.read_csv(y_test_file)

    assert X_train_df.shape == X_train_processed.shape
    assert X_test_df.shape == X_test_processed.shape
    assert len(y_train_df) == len(y_train)
    assert len(y_test_df) == len(y_test)
