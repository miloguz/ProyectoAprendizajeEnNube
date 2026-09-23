"""Orquestación con Prefect del ciclo de vida completo de Machine Learning.

Encadena las etapas: **adquisición de datos** -> **procesamiento** ->
**feature engineering** (embebido en el ``Pipeline`` de sklearn, sin fuga de
datos) -> **entrenamiento y optimización de hiperparámetros** ->
**evaluación** -> **registro y versionado en MLflow**. Al final selecciona
el **modelo candidato** (mayor F1 en test) y lo deja listo para servir
(API / Docker) como ``models/candidate_pipeline.joblib``.

Reutiliza las funciones ya existentes en ``src/data``, ``src/features`` y
``src/models`` — este módulo solo las orquesta.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from src.config.constants import MODELS_DIR, RANDOM_STATE, RAW_DATASET_PATH
from src.config.mlflow_setup import REGISTERED_MODEL_NAME, set_tracking
from src.data.loaders import load_raw_data
from src.features.engineering import split_X_y
from src.models.train import evaluate, get_models, threshold_for_min_precision, tune_model

# logging.getLogger (no get_run_logger de Prefect) para que las tareas sigan
# siendo invocables/testeables fuera de un flow run activo (p.ej. vía .fn en tests).
logger = logging.getLogger(__name__)

# Estrategia de búsqueda por modelo (igual que en notebooks/03_tuning.ipynb):
# malla para los modelos baratos, aleatoria para Random Forest.
SEARCH_STRATEGY = {
    "Logistic Regression": "grid",
    "Decision Tree": "grid",
    "Random Forest": "random",
}


@task(name="acquire-data", retries=2, retry_delay_seconds=5)
def acquire_data(path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Adquisición de datos: valida y carga el dataset crudo desde ``data/raw``."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}. Descárgalo (ver README) y "
            "colócalo en esa ruta antes de correr el pipeline."
        )
    df = load_raw_data(path)
    logger.info("Datos adquiridos: %s filas x %s columnas", df.shape[0], df.shape[1])
    return df


@task(name="process-data")
def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Procesamiento: descarta la columna id y separa features (X) de target (y)."""
    X, y = split_X_y(df)
    logger.info("X: %s, y: %s, positivos: %.1f%%", X.shape, y.shape, 100 * y.mean())
    return X, y


@task(name="split-train-test")
def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado train/test (misma semilla que el resto del proyecto)."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


@task(name="train-and-tune", retries=1)
def train_and_tune(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena y optimiza hiperparámetros con CV, optimizando F1.

    El feature engineering (imputación, escalado, one-hot) va embebido en el
    ``Pipeline`` que ajusta ``tune_model`` — se recalcula en cada fold, sin
    fuga de datos hacia validación/test.
    """
    strategy = SEARCH_STRATEGY[name]
    searcher = tune_model(name, X_train, y_train, search=strategy, scoring="f1", n_iter=15)
    logger.info("%s: F1 (CV) = %.4f | params=%s", name, searcher.best_score_, searcher.best_params_)
    return searcher


@task(name="evaluate-and-log")
def evaluate_and_log(name: str, searcher, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalúa en test y registra el run en MLflow (parámetros, métricas, pipeline)."""
    pipeline = searcher.best_estimator_
    metrics = evaluate(pipeline, X_test, y_test)
    recall_pos = recall_score(y_test, metrics["y_pred"])
    threshold = (
        threshold_for_min_precision(y_test, metrics["y_proba"], min_precision=0.80)
        if metrics["y_proba"] is not None
        else 0.5
    )

    with mlflow.start_run(run_name=f"{name} (orchestrated)") as run:
        mlflow.set_tag("stage", "orchestrated")
        mlflow.log_param("model_name", name)
        mlflow.log_params({f"model__{k}": v for k, v in searcher.best_params_.items()})
        mlflow.log_metrics(
            {
                "f1_cv": searcher.best_score_,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "recall_pos": recall_pos,
                "threshold": threshold,
            }
        )
        mlflow.sklearn.log_model(pipeline, name="pipeline", serialization_format="pickle")
        run_id = run.info.run_id

    return {
        "model_name": name,
        "pipeline": pipeline,
        "run_id": run_id,
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "recall_pos": recall_pos,
        "threshold": threshold,
        "best_params": searcher.best_params_,
    }


@task(name="select-candidate")
def select_candidate(results: list[dict]) -> dict:
    """Selecciona el modelo candidato: el de mayor F1 en test."""
    best = max(results, key=lambda r: r["f1"])
    logger.info("Modelo candidato: %s (F1=%.4f)", best["model_name"], best["f1"])
    return best


@task(name="register-candidate")
def register_candidate(candidate: dict) -> str:
    """Registra el candidato en el Model Registry de MLflow (alias ``candidate``)
    y lo guarda como artefacto local listo para servir (API/Docker)."""
    model_uri = f"runs:/{candidate['run_id']}/pipeline"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "candidate", mv.version)

    MODELS_DIR.mkdir(exist_ok=True)
    artifact_path = MODELS_DIR / "candidate_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": candidate["pipeline"],
            "model_name": candidate["model_name"],
            "best_params": candidate["best_params"],
            "threshold": candidate["threshold"],
            "metrics": {
                "f1": candidate["f1"],
                "roc_auc": candidate["roc_auc"],
                "recall_pos": candidate["recall_pos"],
            },
            "mlflow_run_id": candidate["run_id"],
            "mlflow_model_name": REGISTERED_MODEL_NAME,
            "mlflow_model_version": mv.version,
        },
        artifact_path,
    )

    logger.info(
        "Registrado %s v%s (alias 'candidate') y guardado en %s",
        REGISTERED_MODEL_NAME,
        mv.version,
        artifact_path,
    )
    return artifact_path.as_posix()


@flow(name="student-depression-ml-pipeline", log_prints=True)
def ml_pipeline() -> dict:
    """Orquesta el ciclo de vida completo de ML y devuelve el resumen del candidato.

    Adquisición -> procesamiento -> feature engineering -> entrenamiento y
    optimización -> evaluación -> registro/versionado -> selección del
    modelo candidato.
    """
    set_tracking()

    df = acquire_data()
    X, y = process_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = []
    for name in get_models():
        searcher = train_and_tune(name, X_train, y_train)
        result = evaluate_and_log(name, searcher, X_test, y_test)
        results.append(result)

    candidate = select_candidate(results)
    artifact_path = register_candidate(candidate)

    print(
        f"Modelo candidato: {candidate['model_name']} "
        f"(F1={candidate['f1']:.4f}, ROC-AUC={candidate['roc_auc']:.4f}, "
        f"recall={candidate['recall_pos']:.4f}, umbral={candidate['threshold']:.3f})"
    )
    print(f"Artefacto listo para servir: {artifact_path}")

    return {
        "candidate_model": candidate["model_name"],
        "metrics": {
            "f1": candidate["f1"],
            "roc_auc": candidate["roc_auc"],
            "recall_pos": candidate["recall_pos"],
        },
        "threshold": candidate["threshold"],
        "artifact_path": artifact_path,
    }


if __name__ == "__main__":
    ml_pipeline()
