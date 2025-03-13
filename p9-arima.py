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

# Define the end date for the training set
train_end = '2024-07-31'

# Split the data into training and test sets
train = series[:train_end]
test = series[train_end:]

# Calculate the percentage of data used for training
train_percentage = len(train) / len(series) * 100
test_percentage = len(test) / len(series) * 100

print(f"Training data: {len(train)} samples ({train_percentage:.2f}%)")
print(f"Test data: {len(test)} samples ({test_percentage:.2f}%)")

# Fit the ARIMA model on the training set
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Make predictions for the test set
predictions = model_fit.forecast(steps=len(test))
predictions = pd.Series(predictions, index=test.index)

# Calculate residuals
residuals = test - predictions

# Plotting the results
# plt.figure(figsize=(12, 6))
# plt.plot(train, label='Training Data')
# plt.plot(test, label='Actual Data', color='orange')
# plt.plot(predictions, color='red', label='Predicted Data')
# plt.legend()
# plt.title('ARIMA Forecast for TSLA')
# plt.savefig('p1/fig/arima_forecast_tsla.png')
# plt.show()

# Plotting the residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals', color='purple')
plt.legend()
plt.title('Residuals of ARIMA Forecast for TSLA')
plt.savefig('p1/fig/arima_residuals_tsla.png')
plt.show()