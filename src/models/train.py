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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
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


# ---------------------------------------------------------------------------
# Ajuste de hiperparámetros
# ---------------------------------------------------------------------------
def get_param_grids() -> dict[str, dict]:
    """Rejillas de hiperparámetros para cada modelo de :func:`get_models`.

    Las claves usan el prefijo ``model__`` porque el estimador va dentro del
    Pipeline (paso ``model``), de modo que la búsqueda ajusta el modelo sin
    tocar el preprocesamiento (evita fuga de datos).
    """
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [1, 5, 20],
            "model__criterion": ["gini", "entropy"],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [None, "balanced"],
        },
    }


def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    search: str = "grid",
    scoring: str = "f1",
    cv: int | object = 5,
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbose: int = 0,
):
    """Ajusta los hiperparámetros de un modelo optimizando ``scoring`` (F1 por defecto).

    Args:
        name: Nombre del modelo tal como aparece en :func:`get_models`.
        X_train, y_train: Datos de entrenamiento.
        search: ``"grid"`` (GridSearchCV) o ``"random"`` (RandomizedSearchCV).
        scoring: Métrica a optimizar (por defecto ``"f1"``).
        cv: Nº de folds o un objeto de validación cruzada. Si es un entero, se usa
            :class:`StratifiedKFold` con barajado y semilla fija.
        n_iter: Nº de combinaciones a probar cuando ``search="random"``.
        random_state, n_jobs, verbose: Se propagan al buscador.

    Returns:
        El buscador ya ajustado (``GridSearchCV`` o ``RandomizedSearchCV``), con
        ``best_estimator_``, ``best_params_`` y ``best_score_`` disponibles.
    """
    pipe = build_pipeline(get_models()[name])
    grid = get_param_grids()[name]

    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if search == "random":
        searcher = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            refit=True,
        )
    elif search == "grid":
        searcher = GridSearchCV(
            pipe,
            grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
    else:
        raise ValueError(f"search debe ser 'grid' o 'random', no {search!r}")

    searcher.fit(X_train, y_train)
    return searcher


# ---------------------------------------------------------------------------
# Ajuste del umbral de decisión
# ---------------------------------------------------------------------------
def evaluate_at_threshold(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """Métricas de la clase positiva aplicando un umbral distinto de 0.5.

    Args:
        y_true: Etiquetas reales.
        y_proba: Probabilidad estimada de la clase positiva.
        threshold: Umbral de decisión (``proba >= threshold`` -> clase 1).

    Returns:
        Diccionario con umbral, accuracy, precision, recall y f1.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def threshold_for_min_precision(
    y_true: pd.Series, y_proba: np.ndarray, min_precision: float = 0.80
) -> float:
    """Umbral que maximiza el recall manteniendo la precisión >= ``min_precision``.

    Útil en salud mental: buscamos detectar el mayor número posible de casos
    positivos (recall alto) sin que la precisión caiga por debajo de un mínimo
    aceptable. Si ningún umbral alcanza esa precisión, devuelve el de mayor
    precisión disponible.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve devuelve un elemento más en prec/rec que en thr;
    # el último punto (rec=0) no tiene umbral asociado, así que lo recortamos.
    prec, rec = prec[:-1], rec[:-1]
    mask = prec >= min_precision
    if not mask.any():
        return float(thr[int(np.argmax(prec))])
    candidates = np.where(mask)[0]
    best = candidates[int(np.argmax(rec[candidates]))]
    return float(thr[best])


def best_fbeta_threshold(
    y_true: pd.Series, y_proba: np.ndarray, beta: float = 2.0
) -> float:
    """Umbral que maximiza F-beta. Con ``beta > 1`` se prioriza el recall.

    F2 (beta=2) pondera el recall el doble que la precisión, un criterio
    razonable cuando el coste de un falso negativo es mayor que el de un
    falso positivo.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta**2
    denom = (b2 * prec) + rec
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
    return float(thr[int(np.argmax(fbeta))])
