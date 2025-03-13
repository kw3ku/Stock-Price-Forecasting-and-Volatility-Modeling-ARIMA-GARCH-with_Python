import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Function to parse dates
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

# Load the data
series = pd.read_csv('p1/data/stocks_data.csv', header=[0,1], index_col=0, parse_dates=True)
series = series.squeeze()

print(series.columns)
series = series['Close']['TSLA']

# Ensure the data is numeric
series = pd.to_numeric(series, errors='coerce')

# Drop any rows with NaN values
series = series.dropna()

# Split the data into training and test sets
train_end = '2024-07-31'
train = series[:train_end]
test = series[train_end:]

# Fit the ARIMA model on the training set
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Make predictions for the test set
predictions = model_fit.forecast(steps=len(test))
predictions = pd.Series(predictions, index=test.index)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Data')
plt.plot(predictions, color='red', label='Predicted Data')
plt.legend()
plt.title('ARIMA Forecast for TSLA')
plt.savefig('p1/fig/arima_forecast_tsla.png')
plt.show()