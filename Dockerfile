# Usar uma imagem base do Python
FROM python:3.9-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos necessários
COPY src/ ./src
COPY data/ ./data
COPY requirements.txt ./

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta 8000 para o FastAPI
EXPOSE 8000

# Comando para rodar a aplicação FastAPI
CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
