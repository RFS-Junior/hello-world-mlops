import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Definir o nome do experimento no MLflow
mlflow.set_experiment("Iris_Classification")

# Nome do modelo no Model Registry
MODEL_NAME = "iris_random_forest_model"

# Configuração do MLflow Tracking URI (opcional, se necessário)
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Iniciar o experimento do MLflow
with mlflow.start_run() as run:
    # Carregar os dados
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()

    # Treinar o modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_train)
    accuracy = (y_pred == y_train).mean()

    # Registrar as métricas e parâmetros
    mlflow.log_metric("train_accuracy", accuracy)
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)

    # Salvar o modelo com o MLflow (artefato do run atual)
    artifact_path = "random_forest_model"
    mlflow.sklearn.log_model(model, artifact_path)

    # Registrar o modelo no Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

    try:
        # Criar o registro do modelo, caso ainda não exista
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        # Modelo já registrado - continuar com a criação de uma nova versão
        print(f"O modelo '{MODEL_NAME}' já está registrado no Model Registry.")

    # Criar uma nova versão do modelo no registro
    client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run.info.run_id,
    )

    print(f"Modelo '{MODEL_NAME}' registrado com sucesso no Model Registry!")
