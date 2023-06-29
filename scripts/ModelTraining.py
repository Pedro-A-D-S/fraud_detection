import os
import yaml
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(level=logging.INFO, filename='log/ModelTraining.log',
                    format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

class ModelTraining:
    def __init__(self, X_train_file, X_test_file, y_train_file, y_test_file):
        """
        ModelTraining class for training a random forest model.

        Args:
            X_train_file (str): File path to the training features (X_train).
            X_test_file (str): File path to the testing features (X_test).
            y_train_file (str): File path to the training labels (y_train).
            y_test_file (str): File path to the testing labels (y_test).
        """
        self.X_train_file = X_train_file
        self.X_test_file = X_test_file
        self.y_train_file = y_train_file
        self.y_test_file = y_test_file
        self.param_grid = self.load_config('../configuration/hyperparameters.yaml')

    def load_config(self, config_file: str) -> dict:
        """
        Load the hyperparameter configuration from a YAML file.

        Args:
            config_file (str): File path to the hyperparameter configuration file.

        Returns:
            dict: Loaded hyperparameter configuration.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_file)
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            param_grid = config[1]['grid 2']
            logging.info('Param grid loaded successfully.')
            return param_grid
        except FileNotFoundError:
            logging.error('Hyperparameter configuration file not found.')
            raise
        except Exception as e:
            logging.error(f'Error occurred while loading the hyperparameter configuration: {str(e)}')
            raise

    def load_data(self) -> tuple:
        """
        Load the training and testing data.

        Returns:
            tuple: A tuple containing X_train, X_test, y_train, and y_test as pandas DataFrames.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        X_train_path = os.path.join(script_dir, self.X_train_file)
        X_test_path = os.path.join(script_dir, self.X_test_file)
        y_train_path = os.path.join(script_dir, self.y_train_file)
        y_test_path = os.path.join(script_dir, self.y_test_file)
        try:
            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path)
            y_train = pd.read_csv(y_train_path).squeeze()
            y_test = pd.read_csv(y_test_path).squeeze()
            logging.info('Data loaded successfully.')
            return X_train, X_test, y_train, y_test
        except FileNotFoundError:
            logging.error('Data file not found.')
            raise
        except Exception as e:
            logging.error(f'Error occurred while loading the data: {str(e)}')
            raise

    def perform_hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features as a pandas DataFrame.
            y_train (pd.Series): Training labels as a pandas Series.

        Returns:
            dict: Best hyperparameters found during tuning.
        """
        try:
            rforest = RandomForestClassifier(n_jobs=-1, random_state=42)
            random_search = RandomizedSearchCV(estimator=rforest, param_distributions=self.param_grid,
                                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            logging.info('Hyperparameter tuning completed successfully.')
            return best_params
        except Exception as e:
            logging.error(f'Error occurred during hyperparameter tuning: {str(e)}')
            raise

    def train_model(self, X_train, X_test, y_train, y_test, best_params):
        """
        Train a random forest model using the best hyperparameters.

        Args:
            X_train (pd.DataFrame): Training features as a pandas DataFrame.
            X_test (pd.DataFrame): Testing features as a pandas DataFrame.
            y_train (pd.Series): Training labels as a pandas Series.
            y_test (pd.Series): Testing labels as a pandas Series.
            best_params (dict): Best hyperparameters found during tuning.

        Returns:
            RandomForestClassifier: Trained random forest model.
        """
        try:
            rforest = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            rforest.fit(X_train, y_train)
            y_pred_train = rforest.predict(X_train)
            y_pred_test = rforest.predict(X_test)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
            test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
            logging.info(f"Train Accuracy: {train_accuracy}")
            logging.info(f"Test Accuracy: {test_accuracy}")
            logging.info('Model training completed successfully.')
            return rforest
        except Exception as e:
            logging.error(f'Error occurred during model training: {str(e)}')
            raise

    def save_model(self, model, model_file):
        """
        Save the trained model to a file.

        Args:
            model (RandomForestClassifier): Trained random forest model.
            model_file (str): File path to save the model.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_file)
        try:
            joblib.dump(model, model_path)
            logging.info(f"Model saved successfully at: {model_path}")
        except Exception as e:
            logging.error(f'Error occurred while saving the model: {str(e)}')
            raise

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_train_file = os.path.join(script_dir, 'data/preprocessed/X_train.csv')
    X_test_file = os.path.join(script_dir, 'data/preprocessed/X_test.csv')
    y_train_file = os.path.join(script_dir, 'data/preprocessed/y_train.csv')
    y_test_file = os.path.join(script_dir, 'data/preprocessed/y_test.csv')

    model_file = os.path.join(script_dir, 'model/model_random_forest_2.0.pkl')

    mt = ModelTraining(X_train_file=X_train_file,
                       X_test_file=X_test_file,
                       y_train_file=y_train_file,
                       y_test_file=y_test_file)

    X_train, X_test, y_train, y_test = mt.load_data()

    best_params = mt.perform_hyperparameter_tuning(X_train, y_train)

    model = mt.train_model(X_train, X_test, y_train, y_test, best_params)

    mt.save_model(model, model_file)
