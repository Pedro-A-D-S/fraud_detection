import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level = logging.INFO, filename = 'log/FeatureEngineering.log', format = '%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

class FeatureEngineering:
    def __init__(self, train_file: pd.DataFrame, test_file: pd.DataFrame):
        self.train_file = train_file
        self.test_file = test_file
        
    def load_data(self) -> tuple:
        """
        Loads the train and test files, split them into featuires and labels,
        and returns them as separate variables.
        
        Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
        """
        try:
            df_train = pd.read_csv(self.train_file)
            logging.info('Train file loaded successfully.')
            X_train = df_train.drop('fraude', axis = 1)
            y_train = df_train['fraude']
            logging.info('Train and test defined successfully on train file.')
            
            df_test = pd.read_csv(self.test_file)
            logging.info('Test file loaded successfully.')
            X_test = df_test.drop('fraude', axis = 1)
            y_test = df_test['fraude']
            logging.info('Train and test defined successfully on test file.')
            
            return X_train, X_test, y_train, y_test
        except FileNotFoundError:
            if not df_train:
                logging.error('Train file not found.')
            if not df_test:
                logging.error('Test file not found.')
        except Exception as e:
            logging.error('An error occured while loading the file:' + str(e))
        
    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_features: list, categorical_features: list) -> np.ndarray:
        """
        Preprocesses the input data by applying scaling and encoding transformations.

        Args:
            X_train (pd.DataFrame): The training data features.
            X_test (pd.DataFrame): The test data features.
            numeric_features (list): A list of column names of numeric features.
            categorical_features (list): A list of column names of categorical features.

        Returns:
            np.ndarray: The preprocessed training and test data.

        Raises:
            Exception: If an error occurs during data preprocessing.
        """
        try:
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown = 'ignore')
            preprocessor = ColumnTransformer(
                transformers = [
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])

            X_train_processed = preprocessor.fit_transform(X_train)
            logging.info('X_train was scaled and encoded correctly.')
            X_test_processed = preprocessor.transform(X_test)
            logging.info('X_test was scaled and encoded correctly.')
            return X_train_processed, X_test_processed
        except Exception as e:
            logging.error('An error occurred during data preprocessing.')
            raise e

    
    def save_preprocessed_data(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                X_train_file: str, X_test_file: str, y_train_file: str, y_test_file: str) -> None:
        """
        Saves the preprocessed data to CSV files.

        Args:
            X_train (np.ndarray): The preprocessed training data features.
            X_test (np.ndarray): The preprocessed test data features.
            y_train (np.ndarray): The training data labels.
            y_test (np.ndarray): The test data labels.
            X_train_file (str): The file path to save the preprocessed training data features.
            X_test_file (str): The file path to save the preprocessed test data features.
            y_train_file (str): The file path to save the training data labels.
            y_test_file (str): The file path to save the test data labels.

        Returns:
            None

        Raises:
            Exception: If an error occurs during data saving.

        """
        try:
            # Convert NumPy arrays to pandas DataFrame
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
            y_train_df = pd.DataFrame(y_train)
            y_test_df = pd.DataFrame(y_test)

            # Save DataFrames to CSV files
            X_train_df.to_csv(X_train_file, index = False)
            logging.info(f"Preprocessed training data features saved to: {X_train_file}")
            X_test_df.to_csv(X_test_file, index = False)
            logging.info(f"Preprocessed test data features saved to: {X_test_file}")
            y_train_df.to_csv(y_train_file, index = False)
            logging.info(f"Training data labels saved to: {y_train_file}")
            y_test_df.to_csv(y_test_file, index = False)
            logging.info(f"Test data labels saved to: {y_test_file}")
        except Exception as e:
            logging.error('An error occured during data saving.')
            raise e
        
if __name__ == "__main__":
    train_file = 'data/etl/train.csv'
    test_file = 'data/etl/test.csv'
    numeric_features = ['tempo', 'valor', 'saldo_inicial_c1', 'novo_saldo_c1',
                        'saldo_inicial_c2', 'novo_saldo_c2']
    categorical_features = ['tipo']
    
    
    fe = FeatureEngineering(train_file = train_file,
                            test_file = test_file) 
    X_train, X_test, y_train, y_test = fe.load_data()
    X_train_processed, X_test_processed = fe.preprocess_data(X_train, 
                                                             X_test, 
                                                             numeric_features, 
                                                             categorical_features)
    fe.save_preprocessed_data(X_train_processed, X_test_processed, y_train, y_test,
                              X_train_file = 'data/preprocessed/X_train.csv',
                              X_test_file = 'data/preprocessed/X_test.csv',
                              y_train_file = 'data/preprocessed/y_train.csv',
                              y_test_file = 'data/preprocessed/y_test.csv')
    
    