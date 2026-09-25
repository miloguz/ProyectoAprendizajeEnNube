"""API de predicción para los modelos de riesgo de depresión.

Sirve los pipelines (preprocesador + modelo) guardados por
``src/orchestration/flow.py`` en ``models/model_<slug>.joblib`` — uno por
cada modelo evaluado (Logistic Regression, Decision Tree, Random Forest),
cada uno con su propio umbral de decisión, métricas e hiperparámetros. El
modelo candidato (alias ``MasterModel`` en el MLflow Model Registry) se usa
por defecto en ``POST /predict`` si no se indica otro.

Si no existe ningún ``model_*.joblib`` (artefactos de una corrida más
antigua), cae de vuelta a ``models/candidate_pipeline.joblib`` como único
modelo disponible. El directorio de modelos puede sobreescribirse con la
variable de entorno ``MODEL_PATH`` (ver ``_load_all_models``).

> ⚠️ Nota ética: esta API es una herramienta de análisis estadístico con
> fines académicos, **no** un instrumento de diagnóstico clínico.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictionResponse,
    StudentFeatures,
)
from src.config.constants import MODELS_DIR as DEFAULT_MODELS_DIR

# Artefactos de todos los modelos (uno por modelo evaluado). Si no existen
# (corrida antigua), se cae de vuelta a este único artefacto del candidato.
# Ambas rutas son configurables por entorno (los tests inyectan un directorio
# temporal aislado, sin tocar los artefactos reales del proyecto).
MODEL_GLOB = "model_*.joblib"
MODELS_SCAN_DIR = Path(os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR))
FALLBACK_MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", DEFAULT_MODELS_DIR / "candidate_pipeline.joblib")
)

try:
    # Única fuente de verdad: la versión declarada en pyproject.toml.
    APP_VERSION = pkg_version("student-depression-mlops")
except PackageNotFoundError:
    APP_VERSION = "0.0.0-dev"

model_state: dict[str, Any] = {}


def _load_all_models() -> dict[str, dict]:
    """Carga todos los ``model_*.joblib`` de ``MODELS_SCAN_DIR``, indexados por nombre."""
    models: dict[str, dict] = {}
    for path in sorted(MODELS_SCAN_DIR.glob(MODEL_GLOB)):
        artifact = joblib.load(path)
        models[artifact["model_name"]] = artifact

    if not models and FALLBACK_MODEL_PATH.exists():
        artifact = joblib.load(FALLBACK_MODEL_PATH)
        artifact.setdefault("is_candidate", True)
        artifact.setdefault("n_experiments", 1)
        models[artifact.get("model_name", "unknown")] = artifact

    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = _load_all_models()
    if not models:
        raise RuntimeError(
            f"No se encontraron artefactos de modelo en {MODELS_SCAN_DIR}. Corre el "
            "pipeline de orquestación (uv run python -m src.orchestration.flow) "
            "antes de levantar la API."
        )

    model_state["models"] = models
    model_state["candidate_name"] = next(
        (name for name, a in models.items() if a.get("is_candidate")),
        next(iter(models)),
    )
    yield
    model_state.clear()


app = FastAPI(
    title="Student Depression Risk API",
    description=(
        "Predicción de riesgo de depresión en estudiantes a partir de factores "
        "académicos, de estilo de vida y de salud mental. Herramienta de "
        "análisis estadístico con fines académicos, NO un diagnóstico clínico."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Estado del servicio, versión desplegada y metadatos del modelo candidato."""
    models = model_state.get("models", {})
    candidate_name = model_state.get("candidate_name")
    candidate = models.get(candidate_name) if candidate_name else None
    return HealthResponse(
        status="ok" if models else "model not loaded",
        version=APP_VERSION,
        model_name=candidate_name,
        metrics=candidate.get("metrics") if candidate else None,
        n_models=len(models),
    )


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """Lista los modelos disponibles con sus métricas e hiperparámetros —
    para poblar un selector de predicción o un dashboard comparativo."""
    models = model_state.get("models", {})
    candidate_name = model_state.get("candidate_name", "")
    infos = [
        ModelInfo(
            name=name,
            f1=artifact["metrics"]["f1"],
            roc_auc=artifact["metrics"]["roc_auc"],
            recall_pos=artifact["metrics"]["recall_pos"],
            threshold=float(artifact.get("threshold", 0.5)),
            best_params=artifact.get("best_params", {}),
            n_experiments=int(artifact.get("n_experiments", 1)),
            is_candidate=(name == candidate_name),
        )
        for name, artifact in models.items()
    ]
    return ModelsResponse(models=infos, candidate_model=candidate_name)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures, model_name: str | None = None) -> PredictionResponse:
    """Predice el riesgo de depresión para un estudiante.

    Args:
        features: datos del estudiante.
        model_name: qué modelo usar (ver ``GET /models``). Por defecto, el
            modelo candidato.
    """
    models = model_state.get("models", {})
    if not models:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    name = model_name or model_state.get("candidate_name")
    if name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{name}' no disponible. Consulta GET /models para ver los válidos.",
        )

    artifact = models[name]
    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("threshold", 0.5))

    row = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(proba >= threshold)

    return PredictionResponse(
        depression_risk=prediction,
        probability=proba,
        threshold=threshold,
        model_name=name,
    )
