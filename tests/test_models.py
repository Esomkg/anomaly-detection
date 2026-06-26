import numpy as np
import pytest
import yaml

from src.models.autoencoder import LSTMAutoencoder, SequenceDataset, prepare_sequences
from src.models.ensemble import EnsembleDetector
from src.models.isolation import IsolationForestModel


@pytest.fixture
def config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(500, 10))
    return X


def test_sequence_dataset_length(sample_data):
    dataset = SequenceDataset(sample_data, seq_length=48)
    assert len(dataset) == len(sample_data) - 48 + 1


def test_sequence_dataset_item_shape(sample_data):
    dataset = SequenceDataset(sample_data, seq_length=48)
    x, y = dataset[0]
    assert x.shape == (48, 10)
    assert y.shape == (48, 10)


def test_prepare_sequences_shape(sample_data):
    X, _ = prepare_sequences(sample_data, seq_length=48)
    assert X.shape == (len(sample_data) - 48 + 1, 48, 10)


def test_lstm_autoencoder_forward(config):
    model = LSTMAutoencoder(input_size=10, hidden_size=32, num_layers=1)
    x = torch.randn(4, 48, 10)
    output = model(x)
    assert output.shape == (4, 48, 10)


def test_lstm_autoencoder_reconstruction(config):
    import torch

    model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
    x = torch.randn(2, 48, 5)
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, x)
    assert loss.item() >= 0.0


def test_isolation_forest_fit_predict(config, sample_data):
    model = IsolationForestModel(config)
    model.fit(sample_data)
    scores = model.predict(sample_data)
    assert scores.shape == (len(sample_data),)
    assert scores.dtype == np.float64


def test_isolation_forest_predict_raises_if_not_fitted(config):
    model = IsolationForestModel(config)
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_isolation_forest_save_load(config, sample_data, tmp_path):
    model = IsolationForestModel(config)
    model.fit(sample_data)
    path = tmp_path / "model.pkl"
    model.save(str(path))
    loaded = IsolationForestModel(config)
    loaded.load(str(path))
    scores_orig = model.predict(sample_data)
    scores_loaded = loaded.predict(sample_data)
    np.testing.assert_array_almost_equal(scores_orig, scores_loaded)


def test_ensemble_fit_threshold(config):
    rng = np.random.default_rng(42)
    ae_scores = rng.exponential(1, size=1000)
    if_scores = rng.exponential(1.5, size=1000)
    ensemble = EnsembleDetector(config)
    threshold = ensemble.fit_threshold(ae_scores, if_scores)
    assert isinstance(threshold, float)
    assert 0 < threshold < 10


def test_ensemble_predict(config):
    rng = np.random.default_rng(42)
    ae_scores = rng.exponential(1, size=1000)
    if_scores = rng.exponential(1.5, size=1000)
    ensemble = EnsembleDetector(config)
    predictions, combined = ensemble.predict(ae_scores, if_scores)
    assert predictions.shape == (1000,)
    assert combined.shape == (1000,)
    assert set(predictions).issubset({0, 1})


def test_ensemble_increases_score_for_high_anomaly(config):
    normal_ae = np.array([0.1, 0.2, 0.15])
    normal_if = np.array([0.3, 0.4, 0.35])

    anomaly_ae = np.array([10.0, 12.0, 11.0])
    anomaly_if = np.array([8.0, 9.0, 8.5])

    all_ae = np.concatenate([normal_ae, anomaly_ae])
    all_if = np.concatenate([normal_if, anomaly_if])

    ensemble = EnsembleDetector(config)
    _, combined = ensemble.predict(all_ae, all_if)
    assert combined[3:].mean() > combined[:3].mean()


try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_autoencoder_training(config):
    import pytorch_lightning as pl
    import torch

    model = LSTMAutoencoder(input_size=5, hidden_size=16, num_layers=1, learning_rate=1e-2)
    X = torch.randn(100, 48, 5)
    dataset = torch.utils.data.TensorDataset(X, X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    trainer = pl.Trainer(max_epochs=3, enable_checkpointing=False, logger=False)
    trainer.fit(model, loader)

    output = model(X[:4])
    assert output.shape == (4, 48, 5)
