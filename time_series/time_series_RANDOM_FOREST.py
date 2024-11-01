# Importar bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.preprocessing import StandardScaler

# Configuração de estilo de gráfico
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

# Carregar a base de dados (substitua o caminho do arquivo pelo seu arquivo local)
data = pd.read_csv('./global.csv') 

# Preparar dados: transformar 'Year' em data e definir como índice
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data = data.set_index('Year')
data = data.asfreq('YS')  # Definir frequência anual

# Visualizar os primeiros registros
data.head()

# Plotar os dados para ver o comportamento de 'Total' ao longo do tempo
fig, ax = plt.subplots(figsize=(9, 4))
data['Total'].plot(ax=ax, label='Total CO2 Emissions')
ax.legend()

# Dividir os dados em treino e teste
steps = 30
data_train = data[:-steps].copy()
data_test = data[-steps:].copy()

# Normalizar a série de treino
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train[['Total']])
data_test_scaled = scaler.transform(data_test[['Total']])

# Plotar dados de treino e teste
fig, ax = plt.subplots(figsize=(9, 4))
data_train['Total'].plot(ax=ax, label='train')
data_test['Total'].plot(ax=ax, label='test')
ax.legend()

# Configurar o modelo ForecasterAutoreg com RandomForest
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(max_depth=10, n_estimators=100, random_state=123),
    lags=36  # Definir a quantidade de lags.
)

# Treinar o modelo
forecaster.fit(y=data_train['Total'])

# Fazer previsões para o conjunto de teste
predictions = forecaster.predict(steps=steps)

# Reverter a normalização das previsões
predictions = scaler.inverse_transform(predictions.values.reshape(-1, 1))
predictions_df = pd.DataFrame(predictions, index=data_test.index, columns=['Predictions'])

# Reverter a normalização dos dados de treino e teste para visualização
data_test['Total'] = scaler.inverse_transform(data_test['Total'].values.reshape(-1, 1))
data_train['Total'] = scaler.inverse_transform(data_train['Total'].values.reshape(-1, 1))

# Plotar previsões e comparar com valores reais
fig, ax = plt.subplots(figsize=(9, 4))
data_train['Total'].plot(ax=ax, label='train')
data_test['Total'].plot(ax=ax, label='test')
predictions_df['Predictions'].plot(ax=ax, label='predictions', color='green')
ax.legend()

# Exibir a previsão e a comparação com os valores reais
plt.show()
