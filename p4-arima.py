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
# Split the data into training and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# Fit the ARIMA model and make predictions
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    # Optional: Uncomment the next line to print predicted and actual values
    # print('predicted=%f, expected=%f' % (yhat, obs))

# Plotting the results
plt.plot(test, label='Expected')
plt.plot(predictions, color='red', label='Predicted')
plt.legend()
plt.title('ARIMA Forecast for AAPL')
plt.show()