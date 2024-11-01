# Modelo de Séries Temporais - Machine Learning

**Este diretório contém a implementação de algoritmos de machine learning de séries focados na previsão de níveis globais de CO2 na atmosfera ao longo do tempo(1751 - 2010).**
* Utilizamos uma base de dados disponível no kaggles: https://www.kaggle.com/datasets/programmerrdai/co2-levels-globally-from-fossil-fuels
* Esta base de dados trás os atributos Year, Total, Gas Fuel, Liquid Fuel, Solid Fuel, Cement e Gas Flaring.
* **Variável alvo(target):** Utilizamos o atributo **"Total"** como a variável a ser prevista, representando as emissões de CO2 ao longo do tempo.
* **Variável temporal(feature):** O atributo **"Year"** é usado como a variável temporal que serve como base para a sequência cronológica das previsões.  

#### O projeto explora o uso de modelos de séries temporais como ARIMA e SARIMA para prever emissões futuras, visando:

* Analisar a tendência e sazonalidade das emissões de CO2 ao longo do tempo.
* Implementar técnicas de normalização e transformação para melhorar a qualidade das previsões.
* Comparar o desempenho dos modelos com diferentes combinações de parâmetros e realizar ajuste fino.
* Demonstrar o uso de métricas como RMSE (Root Mean Squared Error) para avaliar a precisão das previsões.
* Aplicar técnicas de backtesting para validar a generalização dos modelos em diferentes períodos históricos.