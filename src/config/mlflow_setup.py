"""Configuración de MLflow para el seguimiento y registro de experimentos.

Usa un backend **local** de dos partes, ambas dentro del proyecto:

- **Tracking store** (runs, params, métricas, *Model Registry*): SQLite en
  ``mlflow.db``. El file store puro (``./mlruns`` como backend) no soporta el
  Model Registry, necesario para versionar el modelo candidato.
- **Artifact store** (pipelines, gráficos): carpeta ``mlruns/`` (como antes).
"""

from __future__ import annotations

import mlflow

from src.config.constants import PROJECT_ROOT

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_LOCATION = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "student-depression"
REGISTERED_MODEL_NAME = "student-depression-classifier"
# Alias que siempre apunta a la versión vigente del modelo candidato en el
# Model Registry (lo que se sirve por API/Docker/Streamlit).
MODEL_ALIAS = "MasterModel"


def set_tracking(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    artifact_location: str = MLFLOW_ARTIFACT_LOCATION,
) -> None:
    """Configura la URI de tracking local y el experimento activo de MLflow.

    Crea el experimento si no existe, fijando su ubicación de artefactos a
    ``artifact_location``; si ya existe, solo lo activa (MLflow ignora
    ``artifact_location`` en ese caso, como es habitual).

    Args:
        tracking_uri: URI del backend de tracking (por defecto, ``sqlite:///mlflow.db``).
        experiment_name: Nombre del experimento a usar/crear.
        artifact_location: Carpeta donde se guardan los artefactos del experimento.
    """
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
