import pytest
import logging
import pandas as pd
from scripts.ETL import ETL

data_file = '../data/raw/fraud_dataset.csv'

# Load the data file as a DataFrame
data = pd.read_csv(data_file)

# Create an instance of the ETL class
etl = ETL()

def test_load_config():
    """
    Test case for the load_config function of the ETL class.

    This test verifies whether the load_config function correctly loads the configuration
    from a given file and ensures that the loaded configuration meets the expected requirements.
    """

    # Provide a sample configuration file path for testing
    config_file = '../configuration/config.yaml'

    # Call the load_config function
    config = etl.load_config(config_file)

    # Perform assertions to validate the loaded configuration
    assert config is not None  # Ensure the configuration is not empty or None
    assert isinstance(config, dict)  # Ensure the configuration is a dictionary
    assert 'column_order' in config  # Ensure 'column_order' key is present in the configuration
    assert 'column_names' in config  # Ensure 'column_names' key is present in the configuration

def test_load_data():
    """
    Test case for the load_data function of the ETL class.

    This test verifies whether the load_data function correctly loads the data from a CSV file
    and returns it as a pandas DataFrame.
    """

    df = etl.load_data(data_file=data_file)

    # Perform assertions to validate the loaded data
    assert df is not None  # Ensure the DataFrame is not empty or None
    assert isinstance(df, pd.DataFrame)  # Ensure the loaded data is a pandas DataFrame
    
    # Test case for loading a non-existent file
    non_existent_file = 'non_existent.csv'
    df_non_existent = etl.load_data(data_file = non_existent_file)
    assert df_non_existent is None
    
    # Test case for loading a file with incorrect format
    incorrect_format_file = 'incorrect_format.parquet'
    df_incorrect_format = etl.load_data(data_file = incorrect_format_file)
    assert df_incorrect_format is None
    
    # Test case for loading an empty file
    empty_file = 'empty_file.csv'
    pd.DataFrame().to_csv(empty_file, index=False)
    df_empty = etl.load_data(data_file=empty_file)
    assert df_empty is not None and (isinstance(df_empty, pd.DataFrame) and df_empty.empty)

def test_reorder_columns():
    """
    Test case for the reorder_columns function of the ETL class.

    This test verifies whether the reorder_columns function correctly reorders the columns of a DataFrame.
    """

    df = etl.reorder_columns(df = data)

    # Perform assertions to validate the reordered DataFrame
    assert df is not None  # Ensure the DataFrame is not empty or None
    assert isinstance(df, pd.DataFrame)  # Ensure the reordered data is a pandas DataFrame
