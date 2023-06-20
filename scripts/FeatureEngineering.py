import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineering:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        
    def load_data(self):
        df = pd.read_csv(self.train_file)
        X_train = df.drop('fraude', axis = 1)
        y_train = df['fraude']
        
        df = pd.read_csv(self.test_file)
        X_test = df.drop('fraude', axis = 1)
        y_test = df['fraude']
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, X_train, X_test, numeric_features, categorical_features):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown = 'ignore')

        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        return X_train_processed, X_test_processed

    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, X_train_file, X_test_file, y_train_file, y_test_file):
        # Convert NumPy arrays to pandas DataFrame
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)

        # Save DataFrames to CSV files
        X_train_df.to_csv(X_train_file, index=False)
        X_test_df.to_csv(X_test_file, index=False)
        y_train_df.to_csv(y_train_file, index=False)
        y_test_df.to_csv(y_test_file, index=False)

        
if __name__ == "__main__":
    train_file = 'data/etl/train.csv'
    test_file = 'data/etl/test.csv'
    numeric_features = ['tempo', 'valor', 'saldo_inicial_c1', 'novo_saldo_c1',
                        'saldo_inicial_c2', 'novo_saldo_c2']
    categorical_features = ['tipo']
    
    
    fe = FeatureEngineering(train_file = train_file,
                            test_file = test_file) 
    X_train, X_test, y_train, y_test = fe.load_data()
    X_train_processed, X_test_processed = fe.preprocess_data(X_train, 
                                                             X_test, 
                                                             numeric_features, 
                                                             categorical_features)
    fe.save_preprocessed_data(X_train_processed, X_test_processed, y_train, y_test,
                              X_train_file = 'data/preprocessed/X_train.csv',
                              X_test_file = 'data/preprocessed/X_test.csv',
                              y_train_file = 'data/preprocessed/y_train.csv',
                              y_test_file = 'data/preprocessed/y_test.csv')
    
    