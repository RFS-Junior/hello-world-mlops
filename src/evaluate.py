import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Carregar o modelo treinado
model = joblib.load('models/random_forest_model.pkl')

# Carregar os dados de teste
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Relatório de Classificação (Test Data):")
print(classification_report(y_test, y_pred))
