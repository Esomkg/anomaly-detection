FROM python:3.11-slim

WORKDIR /app

COPY conda.yaml .
RUN python -c "import yaml; d=yaml.safe_load(open('conda.yaml')); \
    pip_deps=[p for dep in d.get('dependencies',[]) if isinstance(dep,dict) \
    for p in dep.get('pip',[])]; print('\n'.join(pip_deps))" > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
