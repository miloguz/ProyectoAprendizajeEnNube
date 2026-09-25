# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y el proyecto sigue [Semantic Versioning](https://semver.org/lang/es/).

## [0.3.0] - 2026-09-25

### Added

- El pipeline de orquestación guarda **los 3 modelos evaluados** (no solo el
  candidato) como `models/model_<slug>.joblib`, cada uno con su pipeline,
  hiperparámetros, umbral, métricas y nº de experimentos en MLflow
  (`save-model-artifacts`, `src/orchestration/flow.py`).
- La API sirve los 3 modelos: `GET /models` (métricas, hiperparámetros,
  cuál es el candidato) y `POST /predict?model_name=...` para elegir con
  cuál predecir (antes, solo el candidato).
- Streamlit gana dos pestañas: **Predicción** (con selector de los 3
  modelos) y **Dashboard** (tabla comparativa, gráfico de F1/ROC-AUC/recall
  e hiperparámetros por modelo).
- Manual de instalación y uso en PDF (`Manual_Instalacion.pdf`).

### Changed

- Los artefactos `.joblib` del pipeline de orquestación se guardan
  comprimidos (`compress=3`) — el de Random Forest baja de ~150MB a ~26MB.
- Se eliminó el alias `candidate`, huérfano en el MLflow Model Registry
  desde antes de renombrarlo a `MasterModel`.

## [0.2.0] - 2026-09-22

### Added

- Seguimiento de experimentos con **MLflow** (`src/config/mlflow_setup.py`,
  `src/models/train_mlflow.py`, `notebooks/04_mlflow_tracking.ipynb`).
- Backend de tracking en **SQLite** (`mlflow.db`) para habilitar el *Model
  Registry* de MLflow (el file store puro no lo soporta).
- **Orquestación del pipeline de ML con Prefect** (`src/orchestration/flow.py`):
  adquisición de datos, procesamiento, feature engineering, entrenamiento y
  optimización de hiperparámetros, evaluación, y registro/versionado del
  modelo candidato.
- Selección automática del **modelo candidato** (mayor F1 en test) y su
  versionado en el MLflow Model Registry bajo el alias `MasterModel`.
- **API de predicción** con FastAPI (`src/api/`): `GET /health`,
  `POST /predict`, sirviendo el modelo candidato.
- **Despliegue con Docker**: `Dockerfile` para la API, con usuario no-root,
  `HEALTHCHECK` nativo e imagen etiquetada por versión.
- **Interfaz de predicción con Streamlit** (`src/app/streamlit_app.py`):
  formulario que consume `POST /predict` de la API y muestra el veredicto
  de riesgo con su probabilidad y umbral aplicado.
- Tests de orquestación, de la API y de la app de Streamlit
  (`tests/test_orchestration.py`, `tests/test_api.py`,
  `tests/test_streamlit_app.py`).

### Changed

- La versión del proyecto (`pyproject.toml`) pasa a ser la fuente única de
  verdad para la versión reportada por la API (`/health`, título OpenAPI).
- El alias del modelo en el MLflow Model Registry pasa a llamarse
  `MasterModel` (antes `candidate`).

## [0.1.0] - 2026-09-16

### Added

- Scaffolding inicial del proyecto y análisis exploratorio de datos (EDA).
- Preprocesamiento (imputación, escalado, one-hot) y modelado (Logistic
  Regression, Decision Tree, Random Forest); baseline F1=0.868, ROC-AUC=0.918.
- Ajuste de hiperparámetros (`GridSearchCV`/`RandomizedSearchCV`, optimizando
  F1) y ajuste del umbral de decisión (recall) vía curva precision-recall.
- Pruebas unitarias iniciales (`tests/`).
