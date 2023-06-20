import pandas as pd

class ETL:
    def __init__(self, data_file):
        self.data_file = data_file
        self.train = None
        self.test = None
        
    def load_data(self):
        df = pd.read_csv(self.data_file)
        return df
    
    def reorder_columns(self, df):
        columns = ['isFraud', 
         'step', 
         'type', 
         'amount', 
         'oldbalanceOrg', 
         'newbalanceOrig', 
         'oldbalanceDest', 
         'newbalanceDest']
        
        df[columns]
        
        return df
    
    def rename_columns(self, df):
        column_names = {
            'isFraud': 'fraude',
            'step':'tempo',
            'type':'tipo',
            'amount':'valor',
            'oldbalanceOrg':'saldo_inicial_c1',
            'newbalanceOrig':'novo_saldo_c1',
            'oldbalanceDest':'saldo_inicial_c2',
            'newbalanceDest':'novo_saldo_c2',
            }
        
        df = df.rename(columns = column_names)
        return df
    
    def split_data(self, df, train_frac = 0.8, random_state = 42):
        train = df.sample(frac = train_frac, random_state = random_state)
        test = df.drop(train.index)
        self.train = train
        self.test = test
        
        return train, test
    
    def save_data(self, train_file, test_file):
        self.train.to_csv(train_file, index = False)
        self.test.to_csv(test_file, index = False)
        
if __name__ == '__main__':
    
    etl = ETL(data_file = 'data/raw/fraud_dataset.csv')
    df = etl.load_data()
    df_reordered = etl.reorder_columns(df = df)
    df_renamed = etl.rename_columns(df_reordered)
    etl.split_data(df_renamed, train_frac = 0.8)
    etl.save_data(train_file = 'data/etl/train.csv',
                  test_file = 'data/etl/test.csv')