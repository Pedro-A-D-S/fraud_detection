import pandas as pd
import joblib

class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.column_mapping = None

    def load_model(self) -> None:
        with open(self.model_path, 'rb') as f:
            self.model = joblib.load(f)
            self.column_mapping = dict(zip(self.model.feature_names_in_, self.model.feature_names_in_))
            
    def read_data(self, file_path: str) -> pd.DataFrame:
        df =  pd.read_csv(file_path)
        return df

    def predict(self, data: pd.DataFrame) -> pd.Series:
        data_mapped = data.rename(columns = self.column_mapping)
        predictions = self.model.predict(data_mapped)
        predictions = pd.Series(predictions)
        return predictions
    
    def save_predictions(self, predictions: pd.Series, output_file: str) -> None:
        df = pd.DataFrame({'predictions': predictions})
        df.to_csv(output_file, index = False)

if __name__ == "__main__":

    model_path = 'model/model_random_forest_1.0.pkl'
    data_path = 'data/preprocessed/X_test.csv'
    output_file = 'data/predictions.csv'

    predictor = Predictor(model_path)
    predictor.load_model()

    data = predictor.read_data(data_path)
    predictions = predictor.predict(data)
    predictor.save_predictions(predictions, output_file = output_file)