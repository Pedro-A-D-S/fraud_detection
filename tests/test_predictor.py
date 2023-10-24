import pytest
from scripts.Predictor import Predictor


def test_load_model(test_predictor):
    test_predictor.load_model()
    assert test_predictor.model is not None

# Test loading and predicting on test data
def test_load_data_and_predict(test_predictor, test_data):
    test_predictor.load_model()
    data = test_predictor.load_data(test_data)
    predictions = test_predictor.predict(data)
    assert predictions is not None

# Test saving predictions to a CSV file
def test_save_predictions(test_predictor, test_data, tmp_path):
    test_predictor.load_model()
    data = test_predictor.load_data(test_data)
    predictions = test_predictor.predict(data)
    output_file = tmp_path / 'test_predictions.csv'
    test_predictor.save_predictions(predictions, str(output_file))
    assert output_file.exists()