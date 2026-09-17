"""Pruebas de carga de datos."""

import pandas as pd

from src.config.constants import TARGET
from src.data.loaders import load_raw_data

# Dimensiones esperadas del dataset crudo (student_depression.csv).
EXPECTED_SHAPE = (27901, 18)


def test_load_raw_data_returns_dataframe():
    """load_raw_data debe devolver un DataFrame no vacío."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_raw_data_shape():
    """El dataset crudo debe tener el shape esperado."""
    df = load_raw_data()
    assert df.shape == EXPECTED_SHAPE


def test_target_present_and_binary():
    """La columna objetivo existe y es binaria (0/1)."""
    df = load_raw_data()
    assert TARGET in df.columns
    assert set(df[TARGET].unique()) == {0, 1}
