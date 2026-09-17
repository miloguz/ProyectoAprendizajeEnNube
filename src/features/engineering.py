"""Transformaciones y feature engineering.

Este módulo es un punto de partida; se irá completando en la fase de
modelado (encoding de categóricas, escalado de numéricas, etc.).
"""

import pandas as pd


def drop_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Elimina la columna identificadora si existe (no aporta señal predictiva)."""
    return df.drop(columns=[id_col], errors="ignore")
