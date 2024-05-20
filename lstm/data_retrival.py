import yfinance as yf
import pandas as pd
import datetime

tickers = ["GOOGL", "AMZN", "AAPL", "TSLA", "WMT", "MSFT", "META", "COST", "LMT", "NOC", "UNH"]

current_date = datetime.date.today()

for ticker in tickers:
    print("Processing ticker:", ticker)
    data = yf.download(ticker, start="2004-01-01", end=current_date)
    
    # Reset index to make 'Date' a column
    data.reset_index(inplace=True)
    
    # Add 'Ticker' column with ticker value
    data['Symbol'] = ticker
    
    # Reorder the columns
    data = data[['Date', 'Symbol', 'Open', 'Close', 'Low', 'High', 'Volume']]
    
    # Save to CSV
    file_name = f"./data/{ticker}.csv"
    data.to_csv(path_or_buf=file_name, index=False)