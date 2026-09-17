"""Pruebas del preprocesamiento y la separación X / y."""

import pandas as pd
from sklearn.compose import ColumnTransformer

from src.config.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.data.loaders import load_raw_data
from src.features.engineering import build_preprocessor, split_X_y


def test_split_X_y_drops_target_and_id():
    """split_X_y separa el objetivo y elimina la columna 'id'."""
    df = load_raw_data()
    X, y = split_X_y(df)
    assert TARGET not in X.columns
    assert "id" not in X.columns
    assert len(X) == len(y)
    assert y.name == TARGET


def test_build_preprocessor_is_column_transformer():
    """build_preprocessor devuelve un ColumnTransformer con ramas num y cat."""
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)
    branch_names = [name for name, _, _ in pre.transformers]
    assert "num" in branch_names
    assert "cat" in branch_names


def test_preprocessor_fit_transform_no_nans():
    """Ajustado sobre los datos, el preprocesador transforma sin dejar nulos."""
    df = load_raw_data()
    X, _ = split_X_y(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    # Nº de filas intacto; columnas resultantes = numéricas + one-hot.
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert not pd.isna(Xt).any()
