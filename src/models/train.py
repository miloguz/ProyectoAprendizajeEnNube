"""Definición, entrenamiento y evaluación de modelos de clasificación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config.constants import RANDOM_STATE
from src.features.engineering import build_preprocessor


def get_models() -> dict[str, object]:
    """Devuelve un diccionario de modelos base a comparar."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def build_pipeline(model) -> Pipeline:
    """Envuelve un modelo con el preprocesador en un Pipeline de sklearn."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalúa un pipeline entrenado sobre el conjunto de prueba.

    Returns:
        Diccionario con accuracy, f1, roc_auc, matriz de confusión y reporte.
    """
    y_pred = pipeline.predict(X_test)

    # Probabilidad de la clase positiva para ROC-AUC (si el modelo la soporta)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc = np.nan

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=3),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
