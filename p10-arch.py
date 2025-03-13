import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Function to parse dates
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

# Load the data
series = pd.read_csv('data/stocks_data.csv', header=[0,1], index_col=0, parse_dates=True)
series = series.squeeze()

# List of stock symbols
list_stocks = ["TSLA", "PG", "JPM", "NVDA", "PFE"]

def arima_forecast(stock_symbol):
    # Select the 'Close' price of the stock
    stock_series = series['Close'][stock_symbol]

    # Ensure the data is numeric
    stock_series = pd.to_numeric(stock_series, errors='coerce')

    # Drop any rows with NaN values
    stock_series = stock_series.dropna()

    # Split the data into training and test sets
    train_end = '2024-07-31'
    train = stock_series[:train_end]
    test = stock_series[train_end:]

    # Fit the ARIMA model on the training set
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    print(model_fit.summary())

    # Make predictions for the test set
    predictions = model_fit.forecast(steps=len(test))
    predictions = pd.Series(predictions, index=test.index)

    # Calculate residuals
    residuals = test - predictions

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Training Data')
    plt.plot(test, label='Actual Data', color='orange')
    plt.plot(predictions, color='red', label='Predicted Data')
    plt.legend()
    plt.title(f'ARIMA Forecast for {stock_symbol}')
    plt.savefig(f'fig/arima_forecast_{stock_symbol}.png')
    plt.show()

  

# Loop through each stock and generate the plots
for stock in list_stocks:
    arima_forecast(stock)