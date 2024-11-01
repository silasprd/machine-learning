import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
fig, ax = plt.subplots(figsize=(10, 5))
data['Total'].plot(ax=ax, label='Total CO2 Emissions')
ax.legend()

# Split the data in training and test
steps = 30
data_train = data[:-steps].copy()
data_test = data[-steps:].copy()

# Normalize training series
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train[['Total']])
data_test_scaled = scaler.transform(data_test[['Total']])
data_train['Total'] = data_train_scaled
data_test['Total'] = data_test_scaled

# Plot the data to see how 'Total' behaves over time
fig, ax = plt.subplots(figsize=(10, 5))
data_train['Total'].plot(ax=ax, label='train')
data_test['Total'].plot(ax=ax, label='test')
ax.legend()

# Prepare data in the format expected by phophet
data_train.reset_index(inplace=True)
data_train = data_train.rename(columns={'Year': 'ds', 'Total': 'y'})  # Rename columns

# Inicialize and training Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model.fit(data_train)

# Predictions
future = model.make_future_dataframe(periods=steps, freq='YE')
predictions = model.predict(future)

prediction_test = predictions[['ds', 'yhat']].set_index('ds').iloc[-steps:]
prediction_test['yhat'] = scaler.inverse_transform(prediction_test[['yhat']])

# Revert the normalization of training and test data for viewing
data_test['Total'] = scaler.inverse_transform(data_test['Total'].values.reshape(-1, 1))
data_train['y'] = scaler.inverse_transform(data_train['y'].values.reshape(-1, 1))

# Plot predictions and compare with real values
plt.figure(figsize=(10, 5))
plt.plot(data_train['ds'], data_train['y'], label='train')
plt.plot(data_test.index, data_test['Total'], label='test', color='orange')
plt.plot(prediction_test.index, prediction_test['yhat'], label='predictions', color='green')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.legend()
plt.show()

# Display forecast
plt.show()