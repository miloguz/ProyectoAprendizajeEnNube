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
│   ├── 01_eda.ipynb              # Análisis exploratorio de datos
│   ├── 02_modeling.ipynb         # Modelado y evaluación (baseline)
│   ├── 03_tuning.ipynb           # Ajuste de hiperparámetros y del umbral
│   └── 04_mlflow_tracking.ipynb  # Seguimiento de experimentos con MLflow
├── src/
│   ├── config/               # Constantes/rutas (constants.py) y MLflow (mlflow_setup.py)
│   ├── data/                 # Carga (loaders.py) y utilidades (utils.py)
│   ├── features/             # Feature engineering (engineering.py)
│   ├── models/                # Entrenamiento, evaluación y tracking (train.py, train_mlflow.py)
│   ├── orchestration/         # Pipeline de ML orquestado con Prefect (flow.py)
│   ├── api/                   # API de predicción con FastAPI (main.py, schemas.py)
│   └── app/                   # Interfaz de predicción con Streamlit (streamlit_app.py)
├── tests/                    # Pruebas unitarias (datos, features, modelos, orquestación, API, app)
├── models/                   # Artefactos entrenados (.joblib, no versionados en git)
├── Dockerfile                # Imagen de la API de predicción
├── CHANGELOG.md              # Historial de versiones (Keep a Changelog + SemVer)
├── pyproject.toml            # Dependencias y versión del proyecto (gestionado con uv)
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
- [x] Orquestación del pipeline completo (Prefect) — `src/orchestration/flow.py`
- [x] Registro y versionado del modelo candidato (MLflow Model Registry)
- [x] Despliegue como API + Docker — `src/api/`, `Dockerfile`
- [x] Interfaz de predicción (Streamlit) — `src/app/streamlit_app.py`
- [x] Versionado de código y deploy (SemVer, CHANGELOG, imagen Docker versionada)
- [ ] Monitoreo (fuera del alcance de esta entrega)

## Tests

```bash
uv run pytest
```

Cubren la carga de datos (shape esperado), la construcción del preprocesador y del
pipeline, las utilidades de ajuste de umbral, las tareas de orquestación (Prefect, vía
`.fn` para no depender de un servidor corriendo), la API (FastAPI `TestClient`, con un
pipeline pequeño de prueba inyectado por `MODEL_PATH`) y la lógica no visual de la app
de Streamlit (carga de categorías desde el dataset).

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

Los experimentos (modelos base, versiones tuneadas y los del pipeline orquestado) se
registran en [MLflow](https://mlflow.org/) con un backend **local**, dentro del proyecto
(nada se sube al repo — ver `.gitignore`):

- **Tracking store** (runs, params, métricas, *Model Registry*): SQLite en `mlflow.db`.
  El file store puro (`./mlruns` como backend) no soporta el Model Registry, necesario
  para versionar el modelo candidato.
- **Artifact store** (pipelines, gráficos): carpeta `mlruns/`.

La configuración vive en `src/config/mlflow_setup.py` (`set_tracking()`, experimento
`student-depression`, nombre del modelo registrado `student-depression-classifier`).
`src/models/train_mlflow.py` (modelos base + tuneados) y `src/orchestration/flow.py`
(pipeline orquestado, ver abajo) reutilizan esa configuración.

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
`notebooks/04_mlflow_tracking.ipynb`. El pipeline orquestado (`src/orchestration/flow.py`)
también genera runs, además de tunear y registrar el modelo candidato.

### Cómo ver la UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Y abrir **http://localhost:5000** — ahí se comparan runs (parámetros, métricas,
artefactos) y se ve el **Model Registry** con las versiones de
`student-depression-classifier` (alias `MasterModel` en la última registrada por el
pipeline de orquestación).

## Orquestación del pipeline (Prefect)

`src/orchestration/flow.py` encadena con [Prefect](https://www.prefect.io/) **todo el
ciclo de vida de ML** en un único flow (`student-depression-ml-pipeline`), reutilizando
las funciones ya existentes en `src/data`, `src/features` y `src/models` — no las
reimplementa, solo las orquesta:

| Etapa | Tarea (Prefect) | Qué hace |
|-------|------------------|----------|
| Adquisición de datos | `acquire-data` | Valida que exista el CSV crudo y lo carga |
| Procesamiento | `process-data` | Descarta `id`, separa features (X) de target (y) |
| — | `split-train-test` | Split estratificado train/test |
| Feature engineering | *(embebido)* | `ColumnTransformer` dentro del `Pipeline` de cada modelo — se recalcula por fold, sin fuga de datos |
| Entrenamiento y optimización | `train-and-tune` | `GridSearchCV`/`RandomizedSearchCV` por modelo, optimizando F1 |
| Evaluación + registro | `evaluate-and-log` | Evalúa en test y loguea el run en MLflow (parámetros, métricas, pipeline) |
| Selección del candidato | `select-candidate` | Elige el modelo con mayor F1 en test |
| Registro y versionado | `register-candidate` | Lo versiona en el MLflow Model Registry (alias `MasterModel`) y lo guarda como `models/candidate_pipeline.joblib`, listo para servir |

### Cómo correrlo

Requiere un servidor de Prefect local (una sola vez, en otra terminal):

```bash
uv run prefect server start
```

Y luego, en la terminal del proyecto:

```bash
uv run python -m src.orchestration.flow
```

El dashboard de Prefect queda disponible en **http://localhost:4200** (estado de cada
run, logs por tarea, reintentos). Al terminar, imprime el modelo candidato elegido y la
ruta del artefacto listo para servir.

## Modelo candidato

El pipeline de orquestación entrena y tunea los tres modelos (`Logistic Regression`,
`Decision Tree`, `Random Forest`) y elige como **candidato a producción** el de mayor
**F1 en test** — en las corridas realizadas, consistentemente **Logistic Regression**
(`C=0.01`), con:

| Métrica (test) | Valor |
|----------------|:-----:|
| F1 | 0.868 |
| ROC-AUC | 0.919 |
| Recall (clase 1) | 0.881 |
| Umbral aplicado | ~0.33 (recall máx. con precisión ≥ 0.80) |

Se prefiere sobre Random Forest (F1 CV ligeramente menor y sin ventaja práctica) y sobre
Decision Tree (F1 más bajo) por ser, además, el más simple e interpretable de los tres —
relevante en un dominio de salud mental donde explicar una predicción importa. El
artefacto candidato (`models/candidate_pipeline.joblib`) guarda el pipeline **junto con
el umbral de decisión**, para que la regla de decisión viaje con el modelo; también
queda versionado en el MLflow Model Registry (`student-depression-classifier`, alias
`MasterModel`).

**Los otros dos modelos no se descartan:** el flow también guarda cada uno por separado
(`models/model_<slug>.joblib`, con su propio umbral, métricas e hiperparámetros), y la
API los sirve todos (`GET /models`, `POST /predict?model_name=...`) para poder
comparar predicciones entre modelos — visible en el dashboard de Streamlit.

## Despliegue (API + Docker)

Los tres modelos se sirven como un **web-service con API** (FastAPI), empaquetado en
**Docker**. El alcance llega hasta aquí — **sin monitoreo**.

### Localmente

```bash
uv run python -m src.orchestration.flow   # genera models/model_*.joblib (los 3 modelos)
uv run uvicorn src.api.main:app --reload
```

- `GET /health` — estado del servicio, versión y métricas del modelo candidato.
- `GET /models` — lista los 3 modelos disponibles con sus métricas, umbral,
  hiperparámetros y nº de experimentos en MLflow (cuál es el candidato).
- `POST /predict` — recibe las features de un estudiante y devuelve la predicción.
  Acepta un query param opcional `model_name` para elegir el modelo (por defecto,
  el candidato).
- Documentación interactiva (Swagger UI) en **http://localhost:8000/docs**.

Ejemplo:

```bash
curl -X POST "http://localhost:8000/predict?model_name=Decision%20Tree" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male", "Age": 24, "City": "Kalyan", "Profession": "Student",
    "Academic Pressure": 4.0, "Work Pressure": 0.0, "CGPA": 6.5,
    "Study Satisfaction": 2.0, "Job Satisfaction": 0.0,
    "Sleep Duration": "Less than 5 hours", "Dietary Habits": "Unhealthy",
    "Degree": "B.Tech", "Have you ever had suicidal thoughts ?": "Yes",
    "Work/Study Hours": 10.0, "Financial Stress": 5.0,
    "Family History of Mental Illness": "Yes"
  }'
# {"depression_risk":1,"probability":0.99,"threshold":0.37,"model_name":"Decision Tree"}
```

### Con Docker

Los artefactos `models/model_*.joblib` y `models/candidate_pipeline.joblib` **no se
versionan en git**; hay que generarlos antes de construir la imagen (ver el flow de
orquestación arriba).

```bash
docker build -t student-depression-api:0.2.0 -t student-depression-api:latest .
docker run -p 8000:8000 student-depression-api:0.2.0
```

La API queda igual disponible en **http://localhost:8000**. El contenedor corre como
usuario no-root y trae un `HEALTHCHECK` nativo contra `GET /health`
(`docker inspect --format='{{json .State.Health}}' <container>` para verlo).

## Interfaz de predicción (Streamlit)

Además de la API, hay una interfaz gráfica en [Streamlit](https://streamlit.io/)
(`src/app/streamlit_app.py`) pensada para uso no técnico, con dos pestañas:

- **🔮 Predicción:** mismo formulario que la API, más un **selector para elegir
  cuál de los 3 modelos** usar (por defecto, el candidato). Al enviarse llama a
  `POST /predict` y muestra el veredicto (riesgo sí/no, probabilidad, umbral
  aplicado) con la nota ética de siempre — **es una estimación estadística, no un
  diagnóstico clínico**.
- **📊 Dashboard:** tabla comparativa de los 3 modelos (F1, ROC-AUC, recall, umbral,
  nº de experimentos en MLflow, cuál es el candidato) con un gráfico de barras, y
  los hiperparámetros de cada uno — pensado para mostrar en la exposición.

Requiere la API corriendo (local o Docker):

```bash
uv run uvicorn src.api.main:app --reload          # terminal 1
uv run streamlit run src/app/streamlit_app.py     # terminal 2
```

Se abre en **http://localhost:8501**. Si la API corre en otra URL (p. ej. dentro de
Docker con otro host/puerto), se apunta con la variable de entorno:

```bash
STREAMLIT_API_URL=http://localhost:8000 uv run streamlit run src/app/streamlit_app.py
```

## Versionado

El proyecto sigue [Semantic Versioning](https://semver.org/lang/es/) y
[conventional commits](https://www.conventionalcommits.org/), con tres cosas
versionadas de forma trazable entre sí:

- **Código:** la versión vive en `pyproject.toml` (`version`) y queda registrada en
  `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)) por cada
  release; se marcan con un tag de git (`git tag vX.Y.Z`).
- **Deploy (imagen Docker):** se construye etiquetada con la misma versión del proyecto
  (`student-depression-api:0.2.0`, además de `latest`), para saber exactamente qué
  versión del código corre en un contenedor dado. La API expone esa versión en runtime
  vía `GET /health` (leída de `pyproject.toml` con `importlib.metadata`, una única
  fuente de verdad — nunca queda hardcodeada ni desincronizada).
- **Modelo (MLflow Model Registry):** cada corrida del pipeline de orquestación crea una
  nueva versión de `student-depression-classifier` (v1, v2, ...); el alias
  **`MasterModel`** siempre apunta a la versión vigente — la que sirven la API, Docker y
  Streamlit —, sin tener que cambiar código cuando se re-entrena.
