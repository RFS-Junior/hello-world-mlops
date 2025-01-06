FROM python:3.9-slim
WORKDIR /app
COPY src/ ./src
COPY data/ ./data
COPY models/ ./models
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]