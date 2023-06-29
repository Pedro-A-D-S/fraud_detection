import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import scripts.ModelTraining as mt

@pytest.fixture
def model_training_instance():
    X_train_file = "data/preprocessed/X_train.csv"
    X_test_file = "data/preprocessed/X_test.csv"
    y_train_file = "data/preprocessed/y_train.csv"
    y_test_file = "data/preprocessed/y_test.csv"
    return mt.ModelTraining(X_train_file, X_test_file, y_train_file, y_test_file)

def test_load_config(model_training_instance):
    """
    Test case for the load_config method of the ModelTraining class.
    """
    config = model_training_instance.load_config()
    assert isinstance(config, dict)

def test_load_data(model_training_instance):
    """
    Test case for the load_data method of the ModelTraining class.
    """
    X_train, X_test, y_train, y_test = model_training_instance.load_data()
    # Add assertions to check the loaded data

def test_perform_hyperparameter_tuning(model_training_instance):
    """
    Test case for the perform_hyperparameter_tuning method of the ModelTraining class.
    """
    X_train, X_test, y_train, y_test = model_training_instance.load_data()
    best_params = model_training_instance.perform_hyperparameter_tuning(X_train, y_train)
    assert isinstance(best_params, dict)

def test_train_model(model_training_instance):
    """
    Test case for the train_model method of the ModelTraining class.
    """
    X_train, X_test, y_train, y_test = model_training_instance.load_data()
    best_params = model_training_instance.perform_hyperparameter_tuning(X_train, y_train)
    model = model_training_instance.train_model(X_train, X_test, y_train, y_test, best_params)
    # Add assertions to check the trained model

def test_evaluate_model(model_training_instance):
    """
    Test case for the evaluate_model method of the ModelTraining class.
    """
    X_train, X_test, y_train, y_test = model_training_instance.load_data()
    best_params = model_training_instance.perform_hyperparameter_tuning(X_train, y_train)
    model = model_training_instance.train_model(X_train, X_test, y_train, y_test, best_params)
    # Add assertions to check the evaluated model

