import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from pandas import read_csv, to_numeric
from datetime import datetime
from pandas import DataFrame
from matplotlib import pyplot

# List of stock symbols
list_stocks = ["TSLA", "PG", "JPM", "NVDA", "PFE"]

# Get the data for the stock AAPL
data = yf.download(list_stocks, start="2020-01-01", end="2025-02-28")

# Print the first 5 rows of the data
print(data.head())
print(data.tail())

# Save in a csv file in the folder data
# data.to_csv("p1/data/stocks_data.csv")

# ARIMA model
# Load the data
series = read_csv("p1/data/stocks_data.csv", header=[0, 1], index_col=0, parse_dates=True)

# Print the columns of the DataFrame to understand its structure
print(series.columns)

# Select the 'Close' price of the stock 'JPM'
series = series['Close']['JPM']

# Ensure the data is numeric
series = to_numeric(series, errors='coerce')

# Drop any rows with NaN values
series = series.dropna()

print(series.head())
print(series.tail())

# Split the data into training and test sets
train = series[:'2024-06-30']
test = series['2024-07-01':]

# Fit the model on the training set
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Make predictions for the test set
predictions = model_fit.forecast(steps=len(test))
predictions = DataFrame(predictions, index=test.index, columns=['Predicted'])

# Plot the actual and predicted values
pyplot.figure(figsize=(12, 6))
pyplot.plot(train, label='Training Data')
pyplot.plot(test, label='Actual Data')
pyplot.plot(predictions, label='Predicted Data', color='red')
pyplot.legend()
pyplot.show()