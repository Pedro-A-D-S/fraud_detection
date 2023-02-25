import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Read the train and test data into Pandas dataframes
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Separate the target variable from the features in the training data
X_train = train.drop('fraude', axis=1)  # Features
y_train = train['fraude']  # Target variable

# Separate the target variable from the features in the testing data
X_test = test.drop('fraude', axis=1)  # Features
y_test = test['fraude']  # Target variable

# Define the numeric and categorical features to be preprocessed
numeric_features = ['tempo', 'valor', 'saldo_inicial_c1', 'novo_saldo_c1',
       'saldo_inicial_c2', 'novo_saldo_c2']
categorical_features = ['tipo']

# Define the preprocessing pipeline with a ColumnTransformer object
preprocessing_pipeline = ColumnTransformer([
    ('numeric', StandardScaler(), numeric_features), # Scale numeric features
    ('categorical', OneHotEncoder(), categorical_features) # Encode categorical features
])

# Fit the preprocessing pipeline to the training data
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
# Convert the processed data from an array back to a dataframe
X_train_processed = pd.DataFrame(X_train_processed, 
                                columns = numeric_features + 
                                list(preprocessing_pipeline.named_transformers_['categorical'].get_feature_names_out(categorical_features)))

# Apply the same preprocessing to the testing data
X_test_processed = preprocessing_pipeline.transform(X_test)
# Convert the processed data from an array back to a dataframe
X_test_processed = pd.DataFrame(X_test_processed, 
                                columns = numeric_features + 
                                list(preprocessing_pipeline.named_transformers_['categorical'].get_feature_names_out(categorical_features)))

# Save the preprocessed data and target variables as CSV files
X_train_processed.to_csv('../data/preprocessed/X_train.csv', index = False)
X_test_processed.to_csv('../data/preprocessed/X_test.csv', index = False)
y_train.to_csv('../data/preprocessed/y_train.csv', index = False)
y_test.to_csv('../data/preprocessed/y_test.csv', index = False)