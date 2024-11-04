import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

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

# Define and training ARIMA model
model = ARIMA(data_train['Total'], order=(5,2,1))
model_fit = model.fit()

# Predictions
predictions = model_fit.forecast(steps=steps)
predictions = pd.Series(predictions, index=data_test.index)
predictions_df = pd.DataFrame(predictions.values, index=data_test.index, columns=['Predictions'])

# Plot predictions and compare with real values
fig, ax = plt.subplots(figsize=(9, 4))
data_train['Total'].plot(ax=ax, label='train')
data_test['Total'].plot(ax=ax, label='test')
predictions_df['Predictions'].plot(ax=ax, label='predictions', color='green')
ax.legend()

# Display forecast
plt.show()