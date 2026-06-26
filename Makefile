.PHONY: train serve test lint format clean

train:
	python -m src.pipeline.train --config configs/config.yaml

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .hypothesis
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -f mlflow.db
