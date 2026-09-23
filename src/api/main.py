"""API de predicción para el modelo candidato de riesgo de depresión.

Sirve el pipeline (preprocesador + modelo) y el umbral de decisión guardados
en ``models/candidate_pipeline.joblib`` — el artefacto que produce
``src/orchestration/flow.py`` al elegir el modelo candidato. La ruta del
artefacto puede sobreescribirse con la variable de entorno ``MODEL_PATH``.

> ⚠️ Nota ética: esta API es una herramienta de análisis estadístico con
> fines académicos, **no** un instrumento de diagnóstico clínico.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, PredictionResponse, StudentFeatures
from src.config.constants import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "candidate_pipeline.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

model_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el artefacto del modelo en {MODEL_PATH}. Corre el "
            "pipeline de orquestación (uv run python -m src.orchestration.flow) "
            "para generar el modelo candidato antes de levantar la API."
        )
    artifact = joblib.load(MODEL_PATH)
    model_state["pipeline"] = artifact["pipeline"]
    model_state["threshold"] = float(artifact.get("threshold", 0.5))
    model_state["model_name"] = artifact.get("model_name", "unknown")
    model_state["metrics"] = artifact.get("metrics", {})
    yield
    model_state.clear()


app = FastAPI(
    title="Student Depression Risk API",
    description=(
        "Predicción de riesgo de depresión en estudiantes a partir de factores "
        "académicos, de estilo de vida y de salud mental. Herramienta de "
        "análisis estadístico con fines académicos, NO un diagnóstico clínico."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Estado del servicio y metadatos del modelo cargado."""
    loaded = "pipeline" in model_state
    return HealthResponse(
        status="ok" if loaded else "model not loaded",
        model_name=model_state.get("model_name"),
        metrics=model_state.get("metrics"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures) -> PredictionResponse:
    """Predice el riesgo de depresión para un estudiante."""
    if "pipeline" not in model_state:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    pipeline = model_state["pipeline"]
    threshold = model_state["threshold"]

    row = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(proba >= threshold)

    return PredictionResponse(
        depression_risk=prediction,
        probability=proba,
        threshold=threshold,
        model_name=model_state["model_name"],
    )
