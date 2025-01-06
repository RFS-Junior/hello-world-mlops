from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Nome do modelo e estágio do registro
MODEL_NAME = "iris_random_forest_model"  # Certifique-se de usar o nome correto
STAGE = "None"  # Pode ser 'Production', 'Staging', ou 'None' para pegar a última versão

# Carregar o modelo registrado no MLflow
print(f"Carregando o modelo '{MODEL_NAME}' do estágio '{STAGE}'...")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    print(f"Modelo '{MODEL_NAME}' carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    raise e

# Estrutura da requisição
class IrisRequest(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict/")
def predict(request: IrisRequest):
    # Transformar os dados da requisição em um array para predição
    data = np.array([[request.SepalLengthCm, request.SepalWidthCm, request.PetalLengthCm, request.PetalWidthCm]])
    
    # Fazer a predição
    prediction = model.predict(data)
    
    # Mapeamento para os nomes das espécies
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species_map[prediction[0]]
    
    return {"predicted_species": predicted_species}
