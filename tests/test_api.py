from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import src.api.main as api_module
from src.api.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def mock_models(monkeypatch):
    monkeypatch.setattr(api_module, "AUTOENCODER", MagicMock())
    monkeypatch.setattr(api_module, "ISOLATION_FOREST", MagicMock())
    monkeypatch.setattr(api_module, "ENSEMBLE", MagicMock())
    monkeypatch.setattr(api_module, "METADATA", {"feature_cols": [], "input_size": 5})
    monkeypatch.setattr(api_module, "MODEL_CONFIG", {})
    monkeypatch.setattr(api_module, "load_models", lambda *a, **kw: None)


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_health_with_models_loaded(mock_models, client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True


class TestPredictEndpoint:
    valid_payload = {
        "cpu_pct": 50.0,
        "mem_pct": 65.0,
        "disk_io": 30.0,
        "latency_ms": 100.0,
        "error_rate": 0.5,
    }

    @pytest.mark.asyncio
    async def test_predict_returns_200_with_valid_body(self, mock_models, client):
        with patch.object(api_module, "_predict_single", return_value=(False, 0.1)):
            response = await client.post("/predict", json=self.valid_payload)
            assert response.status_code == 200
            data = response.json()
            assert "is_anomaly" in data
            assert "anomaly_score" in data
            assert "threshold" in data

    @pytest.mark.asyncio
    async def test_predict_returns_anomaly_true(self, mock_models, client):
        with patch.object(api_module, "_predict_single", return_value=(True, 0.95)):
            response = await client.post("/predict", json=self.valid_payload)
            data = response.json()
            assert data["is_anomaly"] is True
            assert data["anomaly_score"] == 0.95

    @pytest.mark.asyncio
    async def test_predict_missing_cpu_pct_returns_422(self, mock_models, client):
        payload = {k: v for k, v in self.valid_payload.items() if k != "cpu_pct"}
        response = await client.post("/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_cpu_out_of_range_returns_422(self, mock_models, client):
        payload = {**self.valid_payload, "cpu_pct": 150.0}
        response = await client.post("/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_negative_latency_returns_422(self, mock_models, client):
        payload = {**self.valid_payload, "latency_ms": -5.0}
        response = await client.post("/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_extra_field_accepted(self, mock_models, client):
        with patch.object(api_module, "_predict_single", return_value=(False, 0.1)):
            payload = {**self.valid_payload, "bonus_field": "ignored"}
            response = await client.post("/predict", json=payload)
            assert response.status_code == 200


class TestPredictBatchEndpoint:
    @pytest.mark.asyncio
    async def test_predict_batch_returns_results(self, mock_models, client):
        instances = [
            {
                "cpu_pct": 50.0,
                "mem_pct": 65.0,
                "disk_io": 30.0,
                "latency_ms": 100.0,
                "error_rate": 0.5,
            },
            {
                "cpu_pct": 95.0,
                "mem_pct": 92.0,
                "disk_io": 80.0,
                "latency_ms": 500.0,
                "error_rate": 5.0,
            },
        ]
        with patch.object(
            api_module, "_predict_single", side_effect=[(False, 0.1), (True, 0.9)]
        ):
            response = await client.post("/predict_batch", json={"instances": instances})
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            assert data["predictions"][0]["is_anomaly"] is False
            assert data["predictions"][1]["is_anomaly"] is True

    @pytest.mark.asyncio
    async def test_predict_batch_empty_returns_200(self, mock_models, client):
        response = await client.post("/predict_batch", json={"instances": []})
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == []

    @pytest.mark.asyncio
    async def test_predict_batch_invalid_instance_returns_422(self, mock_models, client):
        instances = [
            {"cpu_pct": 50.0},  # missing required fields
        ]
        response = await client.post("/predict_batch", json={"instances": instances})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_batch_missing_instances_returns_422(self, mock_models, client):
        response = await client.post("/predict_batch", json={})
        assert response.status_code == 422
