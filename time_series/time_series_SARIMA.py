import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Graph style configuration
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

# Load database
data = pd.read_csv('./global.csv')

# Prepare data: transform 'Year' in datetime and define as index
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data = data.set_index('Year')
data = data.asfreq('YS')  # Define anual frequency

# Plot the data to see how 'Total' behaves over time
fig, ax = plt.subplots(figsize=(9, 4))
data['Total'].plot(ax=ax, label='Total CO2 Emissions')
ax.legend()

# Split the data in training and test
steps = 30
data_train = data[:-steps].copy()
data_test = data[-steps:].copy()

# Plot training and test data
fig, ax = plt.subplots(figsize=(9, 4))
data_train['Total'].plot(ax=ax, label='train')
data_test['Total'].plot(ax=ax, label='test')
ax.legend()

# Apply logarithmic transformation to smooth the series
data_train_log = np.log(data_train['Total'])
data_test_log = np.log(data_test['Total'])

# Define and training SARIMA model
# (p, d, q) ARIMA paramaters, e (P, D, Q, S) seasonal parameters
model = SARIMAX(data_train_log, order=(5,2,0), seasonal_order=(1,1,1,10))
model_fit = model.fit()

# Predictions
predictions_log = model_fit.forecast(steps=steps)
predictions = np.exp(predictions_log)  # Reverter a transformação logarítmica

# Convert predictions to DataFrame for viewing
predictions_df = pd.Series(predictions, index=data_test.index, name='Predictions')

# Plot predictions and compare with real values
fig, ax = plt.subplots(figsize=(9, 4))
data_train['Total'].plot(ax=ax, label='train')
data_test['Total'].plot(ax=ax, label='test')
predictions_df.plot(ax=ax, label='predictions', color='green')
ax.legend()

# Display forecast
plt.show()
