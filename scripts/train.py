# data wrangling
import pandas as pd
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# splitting dataframe
x = df_balanced.iloc[:, 1:].values
y = df_balanced.iloc[:, 0].values

# instantiating random forest
rf = RandomForestClassifier(max_depth = 5, random_state = 42)

# hyperparameter tuning

# range for n estimators varying from 20 to 2000 
n_estimators = np.arange(20, 2000, step = 20)
# criterions
criterion = ["gini", "entropy"]
# max_features types
max_features = ["auto", "sqrt", "log2"]
# varying depth of trees from 2 to 200
max_depth = list(np.arange(2, 200, step = 1))
# variying the min samples split from 2 to 100
min_samples_split = np.arange(2, 100, step = 2)
# varying the minime leaves' samples
min_samples_leaf = [1, 2, 4, 6, 8, 10]
# bootstrap true or false
bootstrap = [True, False]

# dictionary with parameters to vary
param_grid = {
    "n_estimators": n_estimators,
    "criterion": criterion,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}