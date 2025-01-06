# Variáveis
DOCKER_IMAGE = hello-world-mlops
DOCKER_CONTAINER = hello-world-mlops-container
DOCKER_PORT = 8000

# 1. Pré-processamento dos dados
preprocess:
	python src/preprocess.py

# 2. Treinamento do modelo
train:
	python src/train.py

# 3. Avaliação do modelo
evaluate:
	python src/evaluate.py

# 4. Deploy do modelo via FastAPI
deploy:
	uvicorn src.deploy:app --reload --host 0.0.0.0 --port 8000

# 5. Rodar a pipeline de ML
ml-pipeline: preprocess train evaluate deploy