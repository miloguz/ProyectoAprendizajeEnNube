"""Utilidades para inspección y resumen de datos."""

import pandas as pd


def dataframe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un resumen por columna: tipo, nulos, % nulos y valores únicos.

    Args:
        df: DataFrame a resumir.

    Returns:
        DataFrame con una fila por columna del original.
    """
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    return summary.sort_values("pct_missing", ascending=False)


def target_balance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Muestra la distribución (conteo y porcentaje) de la variable objetivo.

    Args:
        df: DataFrame de entrada.
        target: Nombre de la columna objetivo.

    Returns:
        DataFrame con conteo y porcentaje por clase.
    """
    counts = df[target].value_counts()
    pct = (df[target].value_counts(normalize=True) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})
