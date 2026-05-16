import yfinance as yf
start="2020-01-01"
end="2025-01-01"

df = yf.download("AAPL",start,end)

print(df.tail())
print(df.shape)
