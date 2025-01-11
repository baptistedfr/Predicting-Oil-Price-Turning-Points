import yfinance as yf

sp500 = yf.Ticker("^GSPC")
data = sp500.history(period="max")
data = data[["Close"]].rename(columns={"Close": "Price"})
data.index = data.index.strftime('%m/%d/%Y')
csv_file_path = "sp500_Price_daily.csv"
data.to_csv(csv_file_path, sep=";")