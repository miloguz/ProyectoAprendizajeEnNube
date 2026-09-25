"""Esquemas Pydantic de entrada/salida de la API de predicción."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StudentFeatures(BaseModel):
    """Features de un estudiante, con los nombres originales del dataset como alias.

    Acepta tanto el nombre "pythónico" del campo (p.ej. ``academic_pressure``)
    como el alias con el nombre original de columna (p.ej. ``"Academic Pressure"``),
    gracias a ``populate_by_name=True``.
    """

    model_config = ConfigDict(populate_by_name=True)

    age: float = Field(alias="Age", ge=0, le=100, examples=[24])
    academic_pressure: float = Field(alias="Academic Pressure", ge=0, le=5, examples=[3.0])
    work_pressure: float = Field(alias="Work Pressure", ge=0, le=5, examples=[0.0])
    cgpa: float = Field(alias="CGPA", ge=0, le=10, examples=[7.5])
    study_satisfaction: float = Field(alias="Study Satisfaction", ge=0, le=5, examples=[3.0])
    job_satisfaction: float = Field(alias="Job Satisfaction", ge=0, le=5, examples=[0.0])
    work_study_hours: float = Field(alias="Work/Study Hours", ge=0, le=24, examples=[6.0])
    financial_stress: float = Field(alias="Financial Stress", ge=0, le=5, examples=[3.0])

    gender: str = Field(alias="Gender", examples=["Male"])
    city: str = Field(alias="City", examples=["Kalyan"])
    profession: str = Field(alias="Profession", examples=["Student"])
    sleep_duration: str = Field(alias="Sleep Duration", examples=["7-8 hours"])
    dietary_habits: str = Field(alias="Dietary Habits", examples=["Moderate"])
    degree: str = Field(alias="Degree", examples=["B.Tech"])
    suicidal_thoughts: str = Field(
        alias="Have you ever had suicidal thoughts ?", examples=["No"]
    )
    family_history_mental_illness: str = Field(
        alias="Family History of Mental Illness", examples=["No"]
    )


class PredictionResponse(BaseModel):
    """Resultado de la predicción para un estudiante."""

    depression_risk: int = Field(description="1 = riesgo detectado, 0 = no detectado")
    probability: float = Field(description="Probabilidad estimada de la clase positiva")
    threshold: float = Field(description="Umbral de decisión aplicado (no necesariamente 0.5)")
    model_name: str = Field(description="Modelo que generó la predicción")


class HealthResponse(BaseModel):
    """Estado del servicio, versión desplegada y del modelo candidato."""

    status: str
    version: str
    model_name: str | None = None
    metrics: dict | None = None
    n_models: int = Field(default=0, description="Cantidad de modelos disponibles para predecir")


class ModelInfo(BaseModel):
    """Métricas e hiperparámetros de un modelo disponible en la API."""

    name: str
    f1: float
    roc_auc: float
    recall_pos: float
    threshold: float = Field(description="Umbral de decisión aplicado por este modelo")
    best_params: dict = Field(description="Hiperparámetros del modelo (tras el tuning)")
    n_experiments: int = Field(description="Nº de runs de MLflow registrados para este modelo")
    is_candidate: bool = Field(description="Si es el modelo candidato (alias MasterModel)")


class ModelsResponse(BaseModel):
    """Listado de modelos disponibles, para poblar selectores y dashboards."""

    models: list[ModelInfo]
    candidate_model: str
