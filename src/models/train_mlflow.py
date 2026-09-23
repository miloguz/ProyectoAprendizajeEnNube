"""Entrenamiento de modelos con seguimiento de experimentos en MLflow.

Reutiliza ``build_pipeline`` y ``evaluate`` de :mod:`src.models.train` para
entrenar cada modelo de :func:`~src.models.train.get_models` (y sus versiones
ya tuneadas en ``notebooks/03_tuning.ipynb``), registrando en MLflow los
parámetros, métricas, el pipeline completo y artefactos gráficos.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, recall_score
from sklearn.model_selection import train_test_split

from src.config.constants import RANDOM_STATE
from src.config.mlflow_setup import set_tracking
from src.data.loaders import load_raw_data
from src.features.engineering import split_X_y
from src.models.train import build_pipeline, evaluate, get_models

# Mejores hiperparámetros encontrados en notebooks/03_tuning.ipynb
# (GridSearchCV / RandomizedSearchCV optimizando F1). Se fijan aquí para poder
# registrar esas versiones ajustadas como runs de MLflow sin repetir la búsqueda.
TUNED_PARAMS: dict[str, dict] = {
    "Logistic Regression": {
        "C": 0.01,
        "class_weight": None,
        "penalty": "l2",
        "solver": "lbfgs",
    },
    "Decision Tree": {
        "class_weight": None,
        "criterion": "entropy",
        "max_depth": 8,
        "min_samples_leaf": 1,
    },
    "Random Forest": {
        "n_estimators": 200,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "max_depth": None,
        "class_weight": None,
    },
}


def _plot_confusion_matrix(y_test, y_pred, title: str, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {title}")
    fig.tight_layout()
    path = tmp_dir / "confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_roc_curve(y_test, y_proba, title: str, tmp_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"Curva ROC — {title}")
    fig.tight_layout()
    path = tmp_dir / "roc_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def train_and_log(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: str | None = None,
    extra_tags: dict | None = None,
) -> str:
    """Entrena un modelo dentro de un Pipeline y registra el run en MLflow.

    Args:
        name: Nombre del modelo tal como aparece en :func:`get_models`.
        model: Estimador sklearn sin ajustar, ya con sus hiperparámetros.
        X_train, y_train, X_test, y_test: Splits de entrenamiento/prueba.
        run_name: Nombre del run en MLflow (por defecto, ``name``).
        extra_tags: Tags adicionales para el run (p.ej. ``{"stage": "tuned"}``).

    Returns:
        El ``run_id`` del run creado.
    """
    run_name = run_name or name
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    recall_pos = recall_score(y_test, metrics["y_pred"])

    with mlflow.start_run(run_name=run_name) as run:
        for tag, value in (extra_tags or {}).items():
            mlflow.set_tag(tag, value)

        mlflow.log_param("model_name", name)
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})

        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "recall_pos": recall_pos,
            }
        )

        mlflow.sklearn.log_model(pipeline, name="pipeline", serialization_format="pickle")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cm_path = _plot_confusion_matrix(y_test, metrics["y_pred"], run_name, tmp_dir)
            mlflow.log_artifact(str(cm_path))
            if metrics["y_proba"] is not None:
                roc_path = _plot_roc_curve(y_test, metrics["y_proba"], run_name, tmp_dir)
                mlflow.log_artifact(str(roc_path))

        return run.info.run_id


def run_all_experiments() -> None:
    """Entrena y registra en MLflow los modelos base y sus versiones tuneadas."""
    set_tracking()

    df = load_raw_data()
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = get_models()

    for name, model in models.items():
        run_id = train_and_log(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            run_name=f"{name} (baseline)",
            extra_tags={"stage": "baseline"},
        )
        print(f"[baseline] {name}: run_id={run_id}")

    for name, model in models.items():
        tuned_model = model.__class__(**{**model.get_params(), **TUNED_PARAMS[name]})
        run_id = train_and_log(
            name,
            tuned_model,
            X_train,
            y_train,
            X_test,
            y_test,
            run_name=f"{name} (tuned)",
            extra_tags={"stage": "tuned"},
        )
        print(f"[tuned]    {name}: run_id={run_id}")


if __name__ == "__main__":
    run_all_experiments()
