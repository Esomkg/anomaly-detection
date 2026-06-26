# Anomaly Detection Pipeline

Production-grade real-time anomaly detection for server metrics using an ensemble of LSTM-Autoencoder and Isolation Forest.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Kafka/Stream │ ──> │ Feature Store │ ──> │ Ensemble Model │ ──> │ Alerting    │
│ (Phase 2)    │     │ (Redis)      │     │ (AE + IF)     │     │ (Grafana)   │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
```

## Tech Stack (Phase 1)

- **Python 3.11** — core language
- **PyTorch + Lightning** — LSTM-Autoencoder
- **scikit-learn** — Isolation Forest
- **FastAPI** — model serving
- **MLflow** — experiment tracking
- **DVC** — data versioning
- **Great Expectations** — data validation
- **pytest + ruff + black** — code quality

## Quick Start

```bash
# Create environment
conda env create -f conda.yaml
conda activate anomaly-detection

# Train pipeline
python -m src.pipeline.train --config configs/config.yaml

# Serve API
uvicorn src.api.main:app --reload

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cpu_pct": 95.0, "mem_pct": 92.0, "disk_io": 45.0, "latency_ms": 200.0, "error_rate": 5.0}'
```

## Project Structure

```
├── configs/            # YAML configuration
├── data/               # raw + processed data
├── models/             # trained model artifacts
├── notebooks/          # EDA and experiments
├── src/
│   ├── data/           # generation + validation
│   ├── features/       # feature engineering
│   ├── models/         # AE, IF, ensemble
│   ├── pipeline/       # training orchestration
│   ├── api/            # FastAPI serving
│   └── utils/          # evaluation metrics
├── tests/              # pytest suite
└── .github/workflows/  # CI
```
