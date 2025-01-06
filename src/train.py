import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Carregar os dados pré-processados
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

# Treinar o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliar o modelo (treinamento)
y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))

# Salvar o modelo treinado
joblib.dump(model, 'models/random_forest_model.pkl')

print("Modelo treinado e salvo com sucesso.")
