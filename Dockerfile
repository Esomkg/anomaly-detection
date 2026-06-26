FROM python:3.11-slim

WORKDIR /app

COPY conda.yaml .
RUN pip install --no-cache-dir -r <(python -c "import yaml; f=open('conda.yaml'); d=yaml.safe_load(f); [print(p) for p in d.get('dependencies', []) if isinstance(p, str) and '=' in p]")

COPY . .

RUN pip install --no-cache-dir pytorch-lightning scikit-learn pandas numpy fastapi uvicorn pydantic mlflow pyyaml joblib kafka-python

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
