# Student Depression — Predicción de Riesgo

Proyecto de especialización en **Ciencia de Datos e IA**. El objetivo es construir un
modelo de **clasificación** que estime el riesgo de depresión en estudiantes a partir
de factores académicos, de estilo de vida y de salud mental.

> ⚠️ **Nota ética:** este dataset contiene información sensible sobre salud mental.
> Se usa exclusivamente con fines académicos. El modelo resultante es una herramienta
> de análisis, **no** un instrumento de diagnóstico clínico.

## Dataset

- **Fuente:** [Student Depression Dataset (Kaggle)](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **Registros:** ~27,900
- **Columnas:** 18
- **Variable objetivo:** `Depression` (0 = no / 1 = sí)

| Grupo | Variables |
|-------|-----------|
| Demográficas | Gender, Age, City, Profession |
| Académicas | Academic Pressure, CGPA, Study Satisfaction, Work/Study Hours, Degree |
| Laborales | Work Pressure, Job Satisfaction |
| Estilo de vida | Sleep Duration, Dietary Habits |
| Salud mental | Have you ever had suicidal thoughts?, Family History of Mental Illness, Financial Stress |

## Estructura del proyecto

```
student-depression-mlops/
├── data/
│   ├── raw/                  # Dataset original (student_depression.csv)
│   └── processed/            # Datos limpios / transformados
├── notebooks/
│   ├── 01_eda.ipynb          # Análisis exploratorio de datos
│   ├── 02_modeling.ipynb     # Modelado y evaluación (baseline)
│   ├── 03_tuning.ipynb       # Ajuste de hiperparámetros y del umbral
│   └── 04_mlflow_tracking.ipynb  # Seguimiento de experimentos con MLflow
├── src/
│   ├── config/               # Constantes/rutas (constants.py) y MLflow (mlflow_setup.py)
│   ├── data/                 # Carga (loaders.py) y utilidades (utils.py)
│   ├── features/             # Feature engineering (engineering.py)
│   └── models/               # Entrenamiento, evaluación y tracking (train.py, train_mlflow.py)
├── tests/                    # Pruebas unitarias
├── pyproject.toml            # Dependencias (gestionadas con uv)
└── .python-version           # Python 3.14
```

## Setup

Este proyecto usa [**uv**](https://docs.astral.sh/uv/) para gestionar el entorno.

```bash
# 1. Crear el entorno virtual e instalar dependencias
uv sync

# 2. Registrar el kernel de Jupyter (para usar en el notebook)
uv run python -m ipykernel install --user --name student-depression

# 3. Abrir el notebook de EDA
uv run jupyter lab notebooks/01_eda.ipynb
```

> Con uv no necesitas activar el entorno manualmente: cualquier comando `uv run ...`
> usa el `.venv` del proyecto.

## Roadmap

- [x] Scaffolding del proyecto
- [x] Descarga del dataset
- [x] Análisis exploratorio (EDA) — `notebooks/01_eda.ipynb`
- [x] Preprocesamiento (imputación + encoding + escalado) — `src/features/engineering.py`
- [x] Modelado (Logistic Regression, Decision Tree, Random Forest) — `src/models/train.py`
- [x] Evaluación (matriz de confusión, ROC-AUC, recall) — `notebooks/02_modeling.ipynb`
- [x] Ajuste de hiperparámetros (GridSearch / RandomizedSearch, optimizando F1) — `notebooks/03_tuning.ipynb`
- [x] Ajuste del umbral de decisión (curva precision-recall, prioriza recall) — `notebooks/03_tuning.ipynb`
- [x] Pruebas unitarias (`uv run pytest`) — `tests/`
- [x] Seguimiento de experimentos (MLflow) — `src/models/train_mlflow.py`, `notebooks/04_mlflow_tracking.ipynb`
- [ ] Despliegue

## Tests

```bash
uv run pytest
```

Cubren la carga de datos (shape esperado), la construcción del preprocesador y del
pipeline, y las utilidades de ajuste de umbral.

## Resultados

### Baseline vs. modelo ajustado (umbral 0.5, test)

Tras el `GridSearchCV`/`RandomizedSearchCV` (optimizando F1), el mejor modelo sigue siendo
**Logistic Regression** (`C=0.01`). El ajuste apenas mueve las métricas frente al baseline
—ya era muy competitivo—, con una ligera mejora en recall:

| Métrica | Baseline (`02`) | Ajustado (`03`) |
|---------|:---------------:|:---------------:|
| Accuracy | 0.844 | 0.844 |
| F1 | 0.868 | 0.869 |
| ROC-AUC | 0.918 | 0.919 |
| Recall (clase 1) | 0.879 | 0.881 |

### Ajuste del umbral de decisión (test)

La mayor ganancia práctica viene de **mover el umbral** por debajo de 0.5 para reducir los
**falsos negativos** (estudiantes en riesgo no detectados), el error más costoso en salud mental.
La curva precision-recall (`03_tuning.ipynb`) hace explícito el equilibrio:

| Estrategia | Umbral | Precisión | Recall | F1 |
|------------|:------:|:---------:|:------:|:--:|
| Por defecto | 0.50 | 0.856 | 0.881 | 0.869 |
| **Precisión ≥ 0.80** (recomendada) | 0.33 | 0.800 | **0.944** | 0.866 |
| Óptimo F2 | 0.18 | 0.741 | 0.978 | 0.843 |

> Con el umbral recomendado (0.33) el recall sube de 0.881 a **0.944** sacrificando solo
> ~0.003 de F1, y los **falsos negativos** en test caen de ~388 a ~184 (**≈ la mitad**).
> El artefacto `models/tuned_pipeline.joblib` guarda el pipeline **junto con el umbral**,
> para que la regla de decisión viaje con el modelo.

> ⚠️ **Nota ética:** priorizar el recall busca no dejar fuera a quien podría necesitar apoyo.
> Aun así, el modelo es una herramienta de **análisis estadístico** con fines académicos,
> **no** un diagnóstico clínico; cualquier uso real exige validación profesional y supervisión humana.

## Seguimiento de experimentos (MLflow)

Los experimentos (modelos base y sus versiones tuneadas) se registran en
[MLflow](https://mlflow.org/) con un backend de tracking **local** (`./mlruns`, no se sube
al repo). La configuración vive en `src/config/mlflow_setup.py` (URI de tracking +
experimento `student-depression`) y el entrenamiento con logging en
`src/models/train_mlflow.py`, que reutiliza `build_pipeline`/`evaluate` de `train.py`.

Para cada modelo se registra en MLflow:

- **Parámetros:** nombre del modelo e hiperparámetros del estimador.
- **Métricas** (test): accuracy, F1, ROC-AUC y recall de la clase positiva.
- **El pipeline completo** (preprocesador + modelo) vía `mlflow.sklearn.log_model`.
- **Artefactos:** matriz de confusión y curva ROC como PNG.

### Cómo generar los runs

```bash
uv run python -m src.models.train_mlflow
```

O de forma interactiva, con comparación de runs incluida, en
`notebooks/04_mlflow_tracking.ipynb`.

### Cómo ver la UI

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Y abrir **http://localhost:5000** — ahí se comparan runs (parámetros, métricas,
artefactos) entre modelos base (`stage=baseline`) y ajustados (`stage=tuned`).

> En Windows, MLflow ≥ 3 exige la variable `MLFLOW_ALLOW_FILE_STORE=true` para seguir
> usando `./mlruns` como backend de archivos (ya se fija automáticamente al importar
> `mlflow_setup.py`). Si se lanza `mlflow ui` desde una terminal nueva y da error de
> backend, fijarla a mano:
>
> ```powershell
> $env:MLFLOW_ALLOW_FILE_STORE = "true"
> uv run mlflow ui --backend-store-uri ./mlruns
> ```
