import pandas as pd
import os

class StockDataFrame:
    script_dir = os.path.abspath('')
    data_dir = f"{script_dir}/lstm/data/" + "{ticker}.csv"
    def __init__(self, ticker):
        self.__ticker = ticker
        self.__ticker_data_path = self.data_dir.format(ticker=self.__ticker.upper())
        self.__stock_df = None
        self.generate_data_frame()
    
    def generate_data_frame(self):
        self.__stock_df = pd.read_csv(self.__ticker_data_path)
        self.__stock_df.set_index('Date', inplace=True)
    
    def get_data_frame(self):
        return self.__stock_df
    
    def get_ticker(self):
        return self.__ticker
    
    def get_ticker_path(self):
        return self.__ticker_data_path

