"""Funciones para cargar el dataset."""

from pathlib import Path

import pandas as pd

from src.config.constants import RAW_DATASET_PATH


def load_raw_data(path: Path | str = RAW_DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset crudo de Student Depression.

    Args:
        path: Ruta al CSV. Por defecto usa el definido en constants.

    Returns:
        DataFrame con los datos crudos.
    """
    df = pd.read_csv(path)
    return df
