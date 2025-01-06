from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Carregar o modelo treinado
model = joblib.load('models/random_forest_model.pkl')

# Inicializar o FastAPI
app = FastAPI()

# Definir a estrutura do corpo da requisição para a previsão
class IrisRequest(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict/")
def predict(request: IrisRequest):
    # Preparar os dados da requisição para fazer a previsão
    data = np.array([[request.SepalLengthCm, request.SepalWidthCm, request.PetalLengthCm, request.PetalWidthCm]])
    
    # Fazer a previsão
    prediction = model.predict(data)    
    
    # Mapear a previsão de volta para o nome da espécie
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_map[prediction[0]]
    
    return {"predicted_species": predicted_species}
