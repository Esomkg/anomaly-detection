import argparse
from pathlib import Path

import mlflow
import numpy as np
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from src.data.generator import generate_metrics
from src.data.validator import validate_metrics
from src.features.builder import build_features
from src.models.autoencoder import LSTMAutoencoder, SequenceDataset, compute_anomaly_scores
from src.models.ensemble import EnsembleDetector
from src.models.isolation import IsolationForestModel
from src.utils.metrics import compute_classification_metrics


def run_pipeline(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment("anomaly-detection")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_n_days": config["data"]["n_days"],
                "ae_hidden_size": config["model"]["autoencoder"]["hidden_size"],
                "ae_sequence_length": config["model"]["autoencoder"]["sequence_length"],
                "if_n_estimators": config["model"]["isolation_forest"]["n_estimators"],
            }
        )

        print("Generating data...")
        df = generate_metrics(config)
        validation = validate_metrics(df)
        if not validation["valid"]:
            print(f"Validation errors: {validation['errors']}")
            return
        mlflow.log_metric("n_samples", len(df))

        print("Building features...")
        feature_df = build_features(df, config)

        label_col = "is_anomaly"
        feature_cols = [c for c in feature_df.columns if c != label_col]
        X = feature_df[feature_cols].values
        y = feature_df[label_col].values

        n = len(X)
        train_end = int(n * config["pipeline"]["train_test_split"])
        val_end = train_end + int(n * config["pipeline"]["validation_split"])

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        _y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        ae_config = config["model"]["autoencoder"]
        seq_length = ae_config["sequence_length"]

        print("Training isolation forest (pre-filter for clean training)...")
        if_model = IsolationForestModel(config)
        if_model.fit(X_train)
        scores_if_train_full = if_model.predict(X_train)

        normal_mask = scores_if_train_full < np.percentile(
            scores_if_train_full,
            (1 - config["data"]["anomaly_rate"]) * 100,
        )
        X_train_clean = X_train[normal_mask]
        print(
            f"Using {len(X_train_clean)}/{len(X_train)} clean points for AE training"
            f" ({len(X_train_clean) / len(X_train) * 100:.1f}%)"
        )

        print("Training autoencoder on clean data...")
        train_dataset = SequenceDataset(X_train_clean, seq_length)
        val_dataset = SequenceDataset(X_val, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=ae_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=ae_config["batch_size"], shuffle=False)

        model = LSTMAutoencoder(
            input_size=X_train_clean.shape[1],
            hidden_size=ae_config["hidden_size"],
            num_layers=ae_config["num_layers"],
            learning_rate=ae_config["learning_rate"],
        )

        trainer = pl.Trainer(
            max_epochs=ae_config["max_epochs"],
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=ae_config["early_stop_patience"],
                )
            ],
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)

        print("Computing autoencoder scores...")
        scores_ae_train = compute_anomaly_scores(model, X_train, seq_length)
        scores_ae_test = compute_anomaly_scores(model, X_test, seq_length)

        scores_if_train = if_model.predict(X_train)
        scores_if_test = if_model.predict(X_test)

        print("Fitting ensemble threshold...")
        ensemble = EnsembleDetector(config)
        ensemble.fit_threshold(scores_ae_train, scores_if_train)
        mlflow.log_metric("ensemble_threshold", ensemble.threshold)

        predictions_train, combined_train = ensemble.predict(scores_ae_train, scores_if_train)
        predictions_test, combined_test = ensemble.predict(scores_ae_test, scores_if_test)

        train_metrics = compute_classification_metrics(
            y_train[: len(predictions_train)],
            predictions_train,
        )
        test_metrics = compute_classification_metrics(
            y_test[: len(predictions_test)],
            predictions_test,
        )

        print(f"Train metrics: {train_metrics}")
        print(f"Test metrics: {test_metrics}")

        mlflow.log_metrics(
            {f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, float)}
        )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, float)}
        )

        artifact_dir = Path("models")
        artifact_dir.mkdir(exist_ok=True)

        trainer.save_checkpoint(str(artifact_dir / "autoencoder.ckpt"))
        if_model.save(str(artifact_dir / "isolation_forest.pkl"))
        mlflow.log_artifact(str(artifact_dir / "isolation_forest.pkl"))

        import json

        metadata = {
            "input_size": X_train.shape[1],
            "feature_cols": feature_cols,
            "ensemble_threshold": ensemble.threshold,
            "ae_sequence_length": seq_length,
            "ae_hidden_size": ae_config["hidden_size"],
            "ae_num_layers": ae_config["num_layers"],
            "ae_learning_rate": ae_config["learning_rate"],
        }
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifacts(str(artifact_dir), artifact_path="models")
        print("Pipeline complete. Run ID:", run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
