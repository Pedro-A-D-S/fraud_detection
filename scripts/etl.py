# data wrangling
import pandas as pd

# reading csv 
df = pd.read_csv('../data/raw/fraud_dataset.csv')

# reordering the columns
df = df[['isFraud', 
         'step', 
         'type', 
         'amount', 
         'oldbalanceOrg', 
         'newbalanceOrig', 
         'oldbalanceDest', 
         'newbalanceDest']]

# creating dictionary containing columns names in order to rename them
colunas = {
    'isFraud': 'fraude',
    'step':'tempo',
    'type':'tipo',
    'amount':'valor',
    'oldbalanceOrg':'saldo_inicial_c1',
    'newbalanceOrig':'novo_saldo_c1',
    'oldbalanceDest':'saldo_inicial_c2',
    'newbalanceDest':'novo_saldo_c2',
}

# renaming columns
df = df.rename(columns = colunas)

# Split the data into 80/20 train/test sets
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

train.to_csv('../data/etl/train.csv', index = False)
test.to_csv('../data/etl/test.csv', index = False)