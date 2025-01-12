import yfinance as yf

sse = yf.Ticker("000001.SS")
data = sse.history(period="max")
data = data[["Close"]].rename(columns={"Close": "Price"})
data.index = data.index.strftime('%m/%d/%Y')
csv_file_path = "SSE_Price_daily.csv"
data.to_csv(csv_file_path, sep=";")