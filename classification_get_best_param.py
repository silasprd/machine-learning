import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

wbc = pd.read_csv('fish_data.csv', encoding='ISO-8859-1')

# Mapping species to numbers
species_mapping = {species: idx for idx, species in enumerate(wbc['species'].unique())}
y = wbc['species'].map(species_mapping)

X = wbc[['length', 'weight']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Instanciates a neural network classifier type with maximum of 2000 epochs 
model = MLPClassifier(max_iter=1)

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (100,100,100), (50,100,50), (100,50,50), (50,50,100), (100,100,50), (50,100,100)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.1],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=5)
clf.fit(X, y)

print('Best parameters found:\n', clf.best_params_)

# # Train model
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Accuracy
# acc = accuracy_score(y_test, y_pred)
# print("Acurácia: {:.2f}".format(acc))

# print('Matriz de Confusão')
# cm = confusion_matrix(y_test, y_pred, labels=y_train.unique())
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_mapping.keys())
# disp.plot()

# plt.xticks(rotation=35, ha='right')
# plt.show()
