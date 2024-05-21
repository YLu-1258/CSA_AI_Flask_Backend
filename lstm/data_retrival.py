import yfinance as yf
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

def generate_data():
    tickers = ["GOOGL", "AMZN", "AAPL", "TSLA", "WMT", "MSFT", "META", "COST", "LMT", "NOC", "UNH"]

    current_date = datetime.date.today()
    start_date = current_date - relativedelta(years=10)

    for ticker in tickers:
        print("Processing ticker:", ticker)
        data = yf.download(ticker, start=start_date, end=current_date)
        
        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)
        
        # Add 'Ticker' column with ticker value
        data['Symbol'] = ticker
        
        # Reorder the columns
        data = data[['Date', 'Symbol', 'Open', 'Close', 'Low', 'High', 'Volume']]
        
        # Save to CSV
        current_directory = Path(__file__).parent.resolve()
        file_name = f"{current_directory}/data/{ticker}.csv"
        data.to_csv(path_or_buf=file_name, index=False)

generate_data()