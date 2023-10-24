import pandas as pd
import pytest

from scripts.etl import ETL
from scripts.FeatureEngineering import FeatureEngineering
from scripts.ModelTraining import ModelTraining


@pytest.fixture(scope="session")
def etl_instance() -> ETL:
    """Returns an instance of the ETL class."""
    return ETL(data_file='data/raw/fraud_dataset.csv')

@pytest.fixture(scope="session")
def etl_instance_empty() -> ETL:
    """Returns an instance of the ETL class."""
    return ETL(data_file='empty_file.csv')

@pytest.fixture(scope="session")
def etl_instance_incorrect_type() -> ETL:
    """Returns an instance of the ETL class."""
    return ETL(data_file='test.parquet')

@pytest.fixture(scope="session")
def etl_instance_empty_file() -> ETL:
    """Returns an instance of the ETL class."""
    return ETL(data_file='empty_file.csv')


@pytest.fixture(scope="session")
def fe_instance(tmp_path) -> FeatureEngineering:
    """Returns an instance of the FeatureEngineering class,
    having been initialized with the given train and test files.

    Args:
        tmp_path: A temporary directory to use for storing the train and test files.

    Returns:
        A FeatureEngineering instance.
    """
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    # Use descriptive variable names and consistent typehints
    train_data = {
        "tempo": [1, 2, 3],
        "valor": [10, 20, 30],
        "saldo_inicial_c1": [100, 200, 300],
        "novo_saldo_c1": [110, 220, 330],
        "saldo_inicial_c2": [1000, 2000, 3000],
        "novo_saldo_c2": [1100, 2200, 3300],
        "tipo": ["A", "B", "C"],
        "fraude": [0, 1, 0],
    }

    test_data = {
        "tempo": [4, 5, 6],
        "valor": [40, 50, 60],
        "saldo_inicial_c1": [400, 500, 600],
        "novo_saldo_c1": [440, 550, 660],
        "saldo_inicial_c2": [4000, 5000, 6000],
        "novo_saldo_c2": [4400, 5500, 6600],
        "tipo": ["D", "E", "F"],
        "fraude": [1, 0, 1],
    }

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    fe = FeatureEngineering(train_file=str(train_file),
                            test_file=str(test_file))

    return fe


@pytest.fixture(scope="session")
def model_training_instance() -> ModelTraining:
    """Returns an instance of the ModelTraining class,
    having been initialized with the given train and test files.

    Args:
        X_train_file: Path to the train features file.
        X_test_file: Path to the test features file.
        y_train_file: Path to the train labels file.
        y_test_file: Path to the test labels file.

    Returns:
        A ModelTraining instance.
    """
    X_train_file = "data/preprocessed/X_train.csv"
    X_test_file = "data/preprocessed/X_test.csv"
    y_train_file = "data/preprocessed/y_train.csv"
    y_test_file = "data/preprocessed/y_test.csv"

    return ModelTraining(X_train_file, X_test_file, y_train_file, y_test_file)
