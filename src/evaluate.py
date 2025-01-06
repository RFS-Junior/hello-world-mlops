import pandas as pd
import mlflow.pyfunc
from sklearn.metrics import accuracy_score

# Configurar o MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Nome do modelo e estágio do registro
MODEL_NAME = "iris_random_forest_model"
STAGE = "None"  # Pode ser "Staging", "Production" ou "None" para a última versão independente do estágio

# Carregar o modelo registrado no MLflow Model Registry
print(f"Carregando o modelo '{MODEL_NAME}' do estágio '{STAGE}'...")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")

# Carregar os dados de teste
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
