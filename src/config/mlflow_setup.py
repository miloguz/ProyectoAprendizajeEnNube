"""Configuración de MLflow para el seguimiento de experimentos."""

from __future__ import annotations

import os

# MLflow >= 3 puso el file store (p.ej. "./mlruns") en modo mantenimiento y
# exige este opt-out explícito para seguir usándolo como backend local.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402 (debe ir después de fijar MLFLOW_ALLOW_FILE_STORE)

from src.config.constants import PROJECT_ROOT  # noqa: E402

MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "student-depression"


def set_tracking(
    tracking_uri: str = MLFLOW_TRACKING_URI, experiment_name: str = EXPERIMENT_NAME
) -> None:
    """Configura la URI de tracking local y el experimento activo de MLflow.

    Args:
        tracking_uri: URI del backend de tracking (por defecto, ``./mlruns``).
        experiment_name: Nombre del experimento a usar/crear.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
