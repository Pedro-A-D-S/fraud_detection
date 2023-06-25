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
        self.param_grid = self.load_config('configuration/hyperparameters.yaml')

    def load_config(self, config_file: str) -> dict:
        """
        Load the hyperparameter configuration from a YAML file.

        Args:
            config_file (str): File path to the hyperparameter configuration file.

        Returns:
            dict: Loaded hyperparameter configuration.
        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            param_grid = config[1]['grid 2']
            logging.info('Param grid loaded successfully.')
            return param_grid
        except FileNotFoundError:
            logging.error('Hyperparameter configuration file not found.')
        except Exception as e:
            logging.error(f'Error occurred while loading the hyperparameter configuration: {str(e)}')

    def load_data(self) -> tuple:
        """
        Load the training and testing data.

        Returns:
            tuple: A tuple containing X_train, X_test, y_train, and y_test as pandas DataFrames.
        """
        try:
            X_train = pd.read_csv(self.X_train_file)
            X_test = pd.read_csv(self.X_test_file)
            y_train = pd.read_csv(self.y_train_file).squeeze()
            y_test = pd.read_csv(self.y_test_file).squeeze()
            logging.info('Data loaded successfully.')
            return X_train, X_test, y_train, y_test
        except FileNotFoundError:
            logging.error('Data file not found.')
        except Exception as e:
            logging.error(f'Error occurred while loading the data: {str(e)}')

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
            rforest = RandomForestClassifier(random_state=75)

            random_cv = RandomizedSearchCV(
                rforest,
                self.param_grid,
                n_iter=10,
                cv=10,
                scoring="recall",
                n_jobs=-1,
                random_state=75
            )
            logging.info('Params loaded successfully.')
            random_cv.fit(X_train, y_train)
            logging.info('Grid fitted successfully.')
            return random_cv.best_params_
        except Exception as e:
            logging.error(f'Error occurred during hyperparameter tuning: {str(e)}')

    def train_model(self, X_train, X_test, y_train, best_params) -> tuple:
        """
        Train the random forest model.

        Args:
            X_train (pd.DataFrame): Training features as a pandas DataFrame.
            X_test (pd.DataFrame): Testing features as a pandas DataFrame.
            y_train (pd.Series): Training labels as a pandas Series.
            best_params (dict): Best hyperparameters found during tuning.

        Returns:
            tuple: A tuple containing the trained model and predicted labels (y_pred).
        """
        try:
            rfn = RandomForestClassifier(**best_params, random_state=75)
            logging.info('Random Forest fitted successfully.')
            model = rfn.fit(X_train, y_train)
            y_pred = rfn.predict(X_test)
            return model, y_pred
        except Exception as e:
            logging.error(f'Error occurred during model training: {str(e)}')

    def save_model(self, model, model_file):
        """
        Save the trained model to a file.

        Args:
            model: Trained model object.
            model_file (str): File path to save the model.
        """
        try:
            joblib.dump(model, model_file)
            logging.info('Model saved successfully.')
        except Exception as e:
            logging.error(f'Error occurred while saving the model: {str(e)}')

    def evaluate_model(self, y_test, y_pred) -> tuple:
        """
        Evaluate the trained model.

        Args:
            y_test (pd.Series): True labels for the testing set.
            y_pred (pd.Series): Predicted labels for the testing set.

        Returns:
            tuple: A tuple containing the accuracy, recall, precision, and F1 score.
        """
        try:
            accuracy = metrics.accuracy_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred)
            return accuracy, recall, precision, f1_score
        except Exception as e:
            logging.error(f'Error occurred while evaluating the model: {str(e)}')

if __name__ == "__main__":
    X_train_file = 'data/preprocessed/X_train.csv'
    X_test_file = 'data/preprocessed/X_test.csv'
    y_train_file = 'data/preprocessed/y_train.csv'
    y_test_file = 'data/preprocessed/y_test.csv'

    model_file = 'model/model_random_forest_2.0.pkl'

    mt = ModelTraining(X_train_file=X_train_file,
                       X_test_file=X_test_file,
                       y_train_file=y_train_file,
                       y_test_file=y_test_file)

    X_train, X_test, y_train, y_test = mt.load_data()

    best_params = mt.perform_hyperparameter_tuning(X_train, y_train)
    if best_params is not None:
        model, y_pred = mt.train_model(X_train, X_test, y_train, best_params)
        if model is not None:
            mt.save_model(model, model_file)
            accuracy, recall, precision, f1_score = mt.evaluate_model(y_test, y_pred)
            logging.info(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1_score:.4f}')
