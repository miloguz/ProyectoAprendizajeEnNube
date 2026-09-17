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
│   └── 01_eda.ipynb          # Análisis exploratorio de datos
├── src/
│   ├── config/               # Constantes y rutas (constants.py)
│   ├── data/                 # Carga (loaders.py) y utilidades (utils.py)
│   ├── features/             # Feature engineering (engineering.py)
│   └── models/               # Entrenamiento y evaluación
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
- [ ] Ajuste de hiperparámetros (GridSearch)
- [ ] Seguimiento de experimentos (MLflow) / despliegue

## Resultados (baseline)

Mejor modelo: **Logistic Regression** (ganador por F1 en validación cruzada).

| Métrica | Valor (test) |
|---------|--------------|
| Accuracy | 0.844 |
| F1 | 0.868 |
| ROC-AUC | 0.918 |
| Recall (clase 1) | 0.879 |

> El recall de la clase positiva (0.879) es clave: minimiza falsos negativos, lo más
> importante en un contexto de salud mental.
