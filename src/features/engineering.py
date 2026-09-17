"""Transformaciones y feature engineering.

Contiene la construcción del preprocesador (ColumnTransformer) que imputa,
escala y codifica las variables para alimentar los modelos.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def drop_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Elimina la columna identificadora si existe (no aporta señal predictiva)."""
    return df.drop(columns=[id_col], errors="ignore")


def split_X_y(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables predictoras (X) de la objetivo (y)."""
    df = drop_id_column(df)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Crea el preprocesador para variables numéricas y categóricas.

    - Numéricas: imputación por mediana + estandarización.
    - Categóricas: imputación por moda + One-Hot Encoding (ignora categorías nuevas).

    Args:
        numeric_features: lista de columnas numéricas (por defecto, la de constants).
        categorical_features: lista de columnas categóricas (por defecto, la de constants).

    Returns:
        ColumnTransformer listo para usar dentro de un Pipeline.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
