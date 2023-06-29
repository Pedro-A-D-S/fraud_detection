import pytest
import logging
import os
import pandas as pd
from scripts.ETL import ETL

data_file = '../data/raw/fraud_dataset.csv'

# Load the data file as a DataFrame
data = pd.read_csv(data_file)

@pytest.fixture
def etl_instance():
    return ETL()

# Create an instance of the ETL class
etl = ETL()

def test_load_config(etl_instance):
    """
    Test case for the load_config function of the ETL class.

    This test verifies whether the load_config function correctly loads the configuration
    from a given file and ensures that the loaded configuration meets the expected requirements.
    """

    # Provide a sample configuration file path for testing
    config_file = '../configuration/config.yaml'

    # Call the load_config function
    config = etl_instance.load_config(config_file)

    # Perform assertions to validate the loaded configuration
    assert config is not None  # Ensure the configuration is not empty or None
    assert isinstance(config, dict)  # Ensure the configuration is a dictionary
    assert 'column_order' in config  # Ensure 'column_order' key is present in the configuration
    assert 'column_names' in config  # Ensure 'column_names' key is present in the configuration

def test_load_csv_data(etl_instance):
    """
    Test case for the load_csv_data function of the ETL class.

    This test verifies whether the load_csv_data function correctly loads the data from a CSV file
    and returns it as a pandas DataFrame.
    """

    # Call the load_csv_data function
    df = etl_instance.load_csv_data(data_file=data_file)

    # Perform assertions to validate the loaded data
    assert df is not None  # Ensure the DataFrame is not empty or None
    assert isinstance(df, pd.DataFrame)  # Ensure the loaded data is a pandas DataFrame
    
    # Test case for loading a non-existent file
    non_existent_file = 'non_existent.csv'
    df_non_existent = etl_instance.load_csv_data(data_file=non_existent_file)
    assert df_non_existent is None
    
    # Test case for loading a file with incorrect format
    incorrect_format_file = 'incorrect_format.parquet'
    df_incorrect_format = etl_instance.load_csv_data(data_file=incorrect_format_file)
    assert df_incorrect_format is None
    
    # Test case for loading an empty file
    empty_file = 'empty_file.csv'
    pd.DataFrame().to_csv(empty_file, index=False)
    df_empty = etl_instance.load_csv_data(data_file=empty_file)
    assert df_empty is None
    
    # Test case for loading a file with missing columns
    missing_columns_file = 'missing_columns.csv'
    df_missing_columns = pd.DataFrame({'col1': [1, 2, 3]})
    df_missing_columns.to_csv(missing_columns_file, index=False)
    df_loaded_missing_columns = etl_instance.load_csv_data(data_file=missing_columns_file)
    expected_columns = ['col1']
    assert set(df_loaded_missing_columns.columns) == set(expected_columns)

def test_reorder_columns_not_empty(etl_instance):
    """
    Test case for the reorder_columns function of the ETL class.

    This test verifies whether the reorder_columns function correctly reorders the columns of a DataFrame.
    """
    data = etl_instance.load_csv_data('../data/raw/fraud_dataset.csv')
    df = etl_instance.reorder_columns(df=data)

    # Perform assertions to validate the reordered DataFrame
    assert df is not None  # Ensure the DataFrame is not empty or None
    assert isinstance(df, pd.DataFrame)  # Ensure the reordered data is a pandas DataFrame

def test_split_data_returns_tuple(etl_instance):
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3],
                       'B': [4, 5, 6],
                       'C': [7, 8, 9]})
    
    # Call the split_data_method
    train, test = etl_instance.split_data(df)
    
    # Assert that the return value is a tuple
    assert isinstance((train, test), tuple)

def test_split_data_returns_dataframes(etl_instance):
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3],
                       'B': [4, 5, 6],
                       'C': [7, 8, 9]})
    
    # Call the split_data method
    train, test = etl_instance.split_data(df)
    
    # Assert that the returned values are DataFrames
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    
def test_split_data_splits_data_properly(etl_instance):
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                       'B': [6, 7, 8, 9, 10]})
    
    # Call the split_data method
    train, test = etl_instance.split_data(df, train_frac=0.6, random_state=42)
    
    # Assert that the training DataFrame has the correct number of rows
    assert train.shape[0] == 3

    # Assert that the test DataFrame has the correct number of rows
    assert test.shape[0] == 2
    
def test_split_datasets_train_test_attributes(etl_instance):
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3],
                       'B': [4, 5, 6],
                       'C': [7, 8, 9]})
    
    # Call the split_data method
    train, test = etl_instance.split_data(df)
    
    # Assert that the ETL instance has the train and test attributes set
    
    assert hasattr(etl_instance, 'train')
    assert hasattr(etl_instance, 'test')
    assert isinstance(etl_instance.train, pd.DataFrame)
    assert isinstance(etl_instance.test, pd.DataFrame)

def test_save_data_saves_files(etl_instance, tmp_path):
    # Create sample training and test DataFrames
    train_df = pd.DataFrame({'A': [1, 2, 3],
                             'B': [4, 5, 6]})
    test_df = pd.DataFrame({'C': [7, 8, 9],
                            'D': [10, 11, 12]})
    
    # Define file paths
    train_file = os.path.join(tmp_path, 'train.csv')
    test_file = os.path.join(tmp_path,'test.csv')
    
    # Set the train and test attributes on the ETL instance
    etl_instance.train = train_df
    etl_instance.test = test_df
    
    # Call the save_data method
    etl_instance.save_data(train_file, test_file)
    
    assert os.path.exists(train_file)
    assert os.path.exists(test_file)

def test_save_data_saves_correct_data(etl_instance, tmp_path):
    # Create sample training and test DataFrames
    train_df = pd.DataFrame({'A': [1, 2, 3],
                             'B': [4, 5, 6]})
    test_df = pd.DataFrame({'C': [7, 8, 9],
                            'D': [10, 11, 12]})
    
    # Define file paths
    train_file = os.path.join(tmp_path, 'train.csv')
    test_file = os.path.join(tmp_path,'test.csv')
    
    # Set the train and test attributes on the ETL instance
    etl_instance.train = train_df
    etl_instance.test = test_df
    
    # Call the save_data method
    etl_instance.save_data(train_file, test_file)
    
    # Load the saved training and test DataFrames
    saved_train_df = pd.read_csv(train_file)
    saved_test_df = pd.read_csv(test_file)
    
    # Assert that the saved DataFrames match the original DataFrames
    assert saved_train_df.equals(train_df)
    assert saved_test_df.equals(test_df)

