import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class ModelTraining:
    def __init__(self, X_train_file, X_test_file, y_train_file, y_test_file):
        self.X_train_file = X_train_file
        self.X_test_file = X_test_file
        self.y_train_file = y_train_file
        self.y_test_file = y_test_file

    def load_data(self):
        X_train = pd.read_csv(self.X_train_file)
        X_test = pd.read_csv(self.X_test_file)
        y_train = pd.read_csv(self.y_train_file).squeeze()
        y_test = pd.read_csv(self.y_test_file).squeeze()
        return X_train, X_test, y_train, y_test

    def hyperparameter_tuning(self, X_train, y_train):
        rforest = RandomForestClassifier(random_state=75)
        n_estimators = np.arange(20, 200, step=20)
        criterion = ["gini", "entropy"]
        max_features = ["sqrt", "log2"]
        max_depth = list(np.arange(2, 20, step=1))
        min_samples_split = np.arange(2, 100, step=2)
        min_samples_leaf = [1, 2, 4, 6, 8, 10]
        bootstrap = [True, False]
        param_grid = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }
        random_cv = RandomizedSearchCV(rforest, param_grid, n_iter=10, cv=10, scoring="recall", n_jobs=-1, random_state=75)
        rcv = random_cv.fit(X_train, y_train)
        return rcv.best_params_

    def train_model(self, X_train, X_test, y_train, best_params):
        rfn = RandomForestClassifier(**best_params, random_state=75)
        model = rfn.fit(X_train, y_train)
        y_pred = rfn.predict(X_test)
        return model, y_pred

    def save_model(self, model, model_file):
        joblib.dump(model, model_file)

    def evaluate_model(self, y_test, y_pred):
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred)
        return accuracy, recall, precision, f1_score

if __name__ == "__main__":
    X_train_file = 'data/preprocessed/X_train.csv'
    X_test_file = 'data/preprocessed/X_test.csv'
    y_train_file = 'data/preprocessed/y_train.csv'
    y_test_file = 'data/preprocessed/y_test.csv'
    
    model_file = '/model/model_random_forest_2.0.pkl'
    
    mt = ModelTraining(X_train_file = X_train_file,
                       X_test_file = X_test_file,
                       y_train_file = y_train_file,
                       y_test_file = y_test_file)
    
    X_train, X_test, y_train, y_test = mt.load_data()
    
    best_params = mt.hyperparameter_tuning(X_train, y_train)
    
    model, y_pred = mt.train_model(X_train, X_test, y_train, best_params)
    
    mt.save_model(model, model_file= model_file)
    
    accuracy, recall, precision, f1_score = mt.evaluate_model(y_test, y_pred)
    