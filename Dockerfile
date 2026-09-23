# Imagen de la API de predicción (student-depression-mlops).
#
# Sirve el modelo candidato (models/candidate_pipeline.joblib) generado por
# el pipeline de orquestación (src/orchestration/flow.py). Ese artefacto NO
# se versiona en git: hay que generarlo antes de construir la imagen.
#
#   uv run prefect server start &
#   uv run python -m src.orchestration.flow
#   docker build -t student-depression-api .
#   docker run -p 8000:8000 student-depression-api

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/

# Capa de dependencias separada del código para aprovechar la caché de Docker.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY models/candidate_pipeline.joblib ./models/candidate_pipeline.joblib

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
