"""Pruebas de construcción de modelos, pipeline y utilidades de umbral."""

import numpy as np
from sklearn.pipeline import Pipeline

from src.models.train import (
    best_fbeta_threshold,
    build_pipeline,
    evaluate_at_threshold,
    get_models,
    get_param_grids,
    threshold_for_min_precision,
)


def test_get_models_keys():
    """get_models expone los tres modelos base esperados."""
    models = get_models()
    assert set(models) == {"Logistic Regression", "Decision Tree", "Random Forest"}


def test_build_pipeline_structure():
    """build_pipeline envuelve el modelo con el preprocesador en un Pipeline."""
    model = get_models()["Logistic Regression"]
    pipe = build_pipeline(model)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["preprocessor", "model"]


def test_param_grids_match_models_and_prefix():
    """Cada modelo tiene rejilla y sus claves usan el prefijo 'model__'."""
    grids = get_param_grids()
    assert set(grids) == set(get_models())
    for grid in grids.values():
        assert all(key.startswith("model__") for key in grid)


def test_evaluate_at_threshold_shifts_predictions():
    """Bajar el umbral aumenta (o mantiene) el recall de la clase positiva."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.2, 0.4, 0.55, 0.6, 0.45, 0.3, 0.7, 0.1])
    base = evaluate_at_threshold(y_true, y_proba, 0.5)
    low = evaluate_at_threshold(y_true, y_proba, 0.3)
    assert set(base) == {"threshold", "accuracy", "precision", "recall", "f1"}
    assert low["recall"] >= base["recall"]


def test_threshold_helpers_return_valid_range():
    """Los umbrales sugeridos caen dentro del rango de probabilidad [0, 1]."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 200), 0, 1)
    t_prec = threshold_for_min_precision(y_true, y_proba, min_precision=0.6)
    t_f2 = best_fbeta_threshold(y_true, y_proba, beta=2.0)
    assert 0.0 <= t_prec <= 1.0
    assert 0.0 <= t_f2 <= 1.0
