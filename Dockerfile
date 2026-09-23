# Imagen de la API de predicción (student-depression-mlops).
#
# Sirve el modelo candidato (models/candidate_pipeline.joblib) generado por
# el pipeline de orquestación (src/orchestration/flow.py). Ese artefacto NO
# se versiona en git: hay que generarlo antes de construir la imagen.
#
#   uv run prefect server start &
#   uv run python -m src.orchestration.flow
#   docker build -t student-depression-api:0.2.0 -t student-depression-api:latest .
#   docker run -p 8000:8000 student-depression-api:0.2.0
#
# El tag de versión (0.2.0) debe coincidir con `version` en pyproject.toml —
# así la imagen desplegada queda trazable a una versión exacta del código
# (ver /health, que la reporta en runtime).

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

# Buena práctica de seguridad: no correr el proceso como root dentro del contenedor.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
