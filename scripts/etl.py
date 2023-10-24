import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, filename = 'log/ETL.log', format = '%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')


class ETL:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.train = None
        self.test = None
        self.column_order = None
        self.column_names = None
        self.load_config('configuration/config.yaml')

    def load_config(self, config_file: str) -> dict:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.column_order = config['column_order']
            self.column_names = config['column_names']
        return config

    def load_data(self) -> pd.DataFrame:
        """
        Loads the data from a CSV file and returns it as a pandas DataFrame.

        Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.data_file)
            logging.info('Data loaded successfully.')
            return df
        except:
            logging.error('It was not possible to load data correctly.')
            return None

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorders the columns of the input DataFrame according to a specified column order and returns the reordered DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The reordered DataFrame.
        """
        try:
            df = df[self.column_order]
            logging.info('Columns reordered successfully.')
            return df
        except:
            logging.error('It was not possible to reorder columns.')
            return None

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the columns of the input DataFrame based on a specified mapping and returns the DataFrame with renamed columns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with renamed columns.
        """
        try:
            df = df.rename(columns = self.column_names)
            logging.info('Columns renamed successfully.')
            return df
        except:
            logging.error('Columns were not renamed successfully.')
            return None

    def split_data(self, df: pd.DataFrame, train_frac: float = 0.8, random_state: int = 42) -> tuple:
        """
        Splits the input DataFrame into training and test sets based on a specified fraction and random state.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            train_frac (float, optional): The fraction of data to be used for training. Defaults to 0.8.
            random_state (int, optional): The random seed for reproducible results. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
        """
        try:
            train = df.sample(frac = train_frac, random_state = random_state)
            test = df.drop(train.index)
            self.train = train
            self.test = test
            logging.info('Data split successfully.')
            return train, test
        except:
            logging.error('Data was not split.')

    def save_data(self, train_file: pd.DataFrame, test_file: pd.DataFrame) -> None:
        """
        Saves the training and test DataFrames to the specified files.

        Args:
            train_file (str): The file path to save the training DataFrame.
            test_file (str): The file path to save the test DataFrame.

        Returns:
            None
        """
        try:
            self.train.to_csv(train_file, index=False)
            self.test.to_csv(test_file, index=False)
            logging.info('Data saved successfully.')
        except:
            logging.error('Data was not saved.')


if __name__ == '__main__':
    etl = ETL(data_file = 'data/raw/fraud_dataset.csv')
    df = etl.load_data()
    df_reordered = etl.reorder_columns(df = df)
    df_renamed = etl.rename_columns(df = df_reordered)
    etl.split_data(df_renamed, train_frac = 0.8)
    etl.save_data(train_file = 'data/etl/train.csv', test_file = 'data/etl/test.csv')
