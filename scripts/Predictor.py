import os
import pickle
import pandas as pd

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)
    
if __name__ == "__main__":

    # Path to the trained model pickle file
    model_path = 'model/model_random_forest_1.0.pkl'
    predictions_folder = 'data/predictions/'
    X_test = pd.read_csv('data/preprocessed/X_test.csv')

    # Example usage
    predictor = Predictor(model_path)
    predictor.load_model()

    # Assuming X_test contains your test dataset
    # Select one registry from X_test for prediction
    test_registry = X_test.iloc[0]  # Change the index as per your requirement

    # Reshape the test registry into a 2D array or dataframe
    test_data = test_registry.values.reshape(1, -1)

    # Make the prediction
    prediction = predictor.predict(test_data)

    # Create the feature_store folder if it doesn't exist
    os.makedirs(predictions_folder, exist_ok = True)

    # Create a DataFrame to store the predictions
    predictions_df = pd.DataFrame({'Prediction': prediction})

    # Save the predictions as a CSV file
    predictions_file = os.path.join(predictions_folder, 'predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)

    print("Predictions saved in:", predictions_file)
