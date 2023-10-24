import os
import pytest
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

def test_load_config(model_training_instance):
    """
    Test case for the load_config method of the ModelTraining class.
    """
    config = model_training_instance.load_config(config_file='configuration/hyperparameters.yaml')
    assert isinstance(config, dict)

def test_load_data(model_training_instance):
    """
    Test case for the load_data method of the ModelTraining class.
    """
    X_train, X_test, y_train, y_test = model_training_instance.load_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

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
    model, y_pred = model_training_instance.train_model(X_train, X_test, y_train, best_params)
    assert model is not None
    assert y_pred is not None