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

# 5. Construção da imagem Docker
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# 6. Execução do contêiner Docker
docker-run:
	docker run -it --rm -p $(DOCKER_PORT):$(DOCKER_PORT) -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models $(DOCKER_IMAGE)

# 7. Rodar a pipeline de ML
ml-pipeline: preprocess train evaluate deploy

# 8. Rodar tudo com Docker
docker-pipeline: preprocess train evaluate docker-build docker-run

