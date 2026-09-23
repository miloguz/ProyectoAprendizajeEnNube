"""Pruebas de las tareas de orquestación (Prefect).

Llaman a la función subyacente de cada tarea (``.fn``) para probar la lógica
sin necesitar un servidor de Prefect corriendo.
"""

import pandas as pd
import pytest

from src.orchestration.flow import acquire_data, process_data, select_candidate
from src.config.constants import RAW_DATASET_PATH, TARGET


def test_acquire_data_loads_expected_shape():
    df = acquire_data.fn(RAW_DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert TARGET in df.columns
    assert df.shape[0] > 0


def test_acquire_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        acquire_data.fn("no/existe/dataset.csv")


def test_process_data_splits_X_y():
    df = acquire_data.fn(RAW_DATASET_PATH)
    X, y = process_data.fn(df)
    assert TARGET not in X.columns
    assert "id" not in X.columns
    assert len(X) == len(y)


def test_select_candidate_picks_highest_f1():
    results = [
        {"model_name": "a", "f1": 0.70},
        {"model_name": "b", "f1": 0.90},
        {"model_name": "c", "f1": 0.85},
    ]
    best = select_candidate.fn(results)
    assert best["model_name"] == "b"
