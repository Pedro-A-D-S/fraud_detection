import pandas as pd
import logging
import joblib

logging.basicConfig(level=logging.INFO, filename='log/Predictor.log',
                    format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

class Predictor:
    def __init__(self, model_path: str):
        """
        Predictor class to load a trained model and make predictions.

        Args:
            model_path (str): File path to the trained model.
        """
        self.model_path = model_path
        self.model = None
        self.column_mapping = None

    def load_model(self) -> None:
        """
        Load the trained model from the model file.
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.model = joblib.load(f)
                self.column_mapping = dict(zip(self.model.feature_names_in_, self.model.feature_names_in_))
            logging.info('Model loaded successfully.')
        except FileNotFoundError:
            logging.error('Model file not found.')
        except Exception as e:
            logging.error(f'Error occurred while loading the model: {str(e)}')

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the data from a CSV file.

        Args:
            file_path (str): File path to the data file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            logging.info('Data loaded successfully.')
            return df
        except FileNotFoundError:
            logging.error('Data file not found.')
        except Exception as e:
            logging.error(f'Error occurred while loading the data: {str(e)}')

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions on the provided data.

        Args:
            data (pd.DataFrame): Input data as a pandas DataFrame.

        Returns:
            pd.Series: Predicted values as a pandas Series.
        """
        try:
            data_mapped = data.rename(columns=self.column_mapping)
            predictions = self.model.predict(data_mapped)
            predictions = pd.Series(predictions)
            logging.info('Predictions successfully done.')
            return predictions
        except Exception as e:
            logging.error(f'Error occurred while making predictions: {str(e)}')

    def save_predictions(self, predictions: pd.Series, output_file: str) -> None:
        """
        Save the predictions to a CSV file.

        Args:
            predictions (pd.Series): Predicted values as a pandas Series.
            output_file (str): File path to save the predictions.
        """
        try:
            df = pd.DataFrame({'predictions': predictions})
            df.to_csv(output_file, index=False)
            logging.info('Predictions correctly saved.')
        except Exception as e:
            logging.error(f'Error occurred while saving the predictions: {str(e)}')

if __name__ == "__main__":
    model_path = 'model/model_random_forest_2.0.pkl'
    data_path = 'data/preprocessed/X_test.csv'
    output_file = 'data/predictions/predictions.csv'

    predictor = Predictor(model_path)
    predictor.load_model()

    data = predictor.load_data(data_path)
    if data is not None:
        predictions = predictor.predict(data)
        if predictions is not None:
            predictor.save_predictions(predictions, output_file=output_file)
