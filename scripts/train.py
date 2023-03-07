import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Load preprocessed datasets
X_train = pd.read_csv('../data/preprocessed/X_train.csv')
X_test = pd.read_csv('../data/preprocessed/X_test.csv')
y_train = pd.read_csv('../data/preprocessed/y_train.csv')
y_test = pd.read_csv('../data/preprocessed/y_test.csv')

# Instantiate a random forest classifier
rforest = RandomForestClassifier(random_state=75)

# Define the range of hyperparameters to test
n_estimators = np.arange(20, 200, step=20) # Number of trees in the forest
criterion = ["gini", "entropy"] # Function to measure the quality of a split
max_features = ["auto", "sqrt", "log2"] # Number of features to consider when looking for the best split
max_depth = list(np.arange(2, 20, step=1)) # Maximum depth of the tree
min_samples_split = np.arange(2, 100, step=2) # Minimum number of samples required to split an internal node
min_samples_leaf = [1, 2, 4, 6, 8, 10] # Minimum number of samples required to be at a leaf node
bootstrap = [True, False] # Whether bootstrap samples are used when building trees

# Create a dictionary with hyperparameters to test
param_grid = {
    "n_estimators": n_estimators,
    "criterion": criterion,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

# Set up a RandomizedSearchCV object to find the best hyperparameters
random_cv = RandomizedSearchCV(rforest,
                               param_grid,
                               n_iter=10,
                               cv=10,
                               scoring="recall",
                               n_jobs=-1,
                               random_state=75)

# Fit the RandomizedSearchCV object to the training data
rcv = random_cv.fit(X_train, y_train)

# Create a dataframe with the results of the hyperparameter tuning
tuning = pd.DataFrame(rcv.cv_results_)

# Instantiate a random forest classifier with the best hyperparameters found
rfn = RandomForestClassifier(**rcv.best_params_, random_state=75)

# Train the model on the training data
model = rfn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfn.predict(X_test)

# Save the trained model to a file
joblib.dump(model, '../model/model_random_forest_1.0.pkl')