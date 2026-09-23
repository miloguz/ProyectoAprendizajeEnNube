"""Pruebas de la API de predicción (FastAPI).

Entrena un pipeline pequeño sobre una muestra del dataset y lo guarda como el
artefacto que la API espera (mismo formato que produce
``src/orchestration/flow.py``), para no depender de haber corrido el
pipeline de orquestación completo antes de testear.
"""

import os

import joblib
import pytest
from fastapi.testclient import TestClient

from src.data.loaders import load_raw_data
from src.features.engineering import split_X_y
from src.models.train import build_pipeline, get_models


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    df = load_raw_data()
    X, y = split_X_y(df)
    sample = X.sample(n=200, random_state=0)
    sample_y = y.loc[sample.index]

    pipeline = build_pipeline(get_models()["Logistic Regression"])
    pipeline.fit(sample, sample_y)

    model_path = tmp_path_factory.mktemp("model") / "candidate_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": "Logistic Regression",
            "threshold": 0.5,
            "metrics": {"f1": 0.0},
        },
        model_path,
    )

    previous = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = str(model_path)

    from src.api.main import app as fastapi_app

    yield fastapi_app

    if previous is None:
        os.environ.pop("MODEL_PATH", None)
    else:
        os.environ["MODEL_PATH"] = previous


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


VALID_PAYLOAD = {
    "Gender": "Male",
    "Age": 24,
    "City": "Kalyan",
    "Profession": "Student",
    "Academic Pressure": 4.0,
    "Work Pressure": 0.0,
    "CGPA": 6.5,
    "Study Satisfaction": 2.0,
    "Job Satisfaction": 0.0,
    "Sleep Duration": "Less than 5 hours",
    "Dietary Habits": "Unhealthy",
    "Degree": "B.Tech",
    "Have you ever had suicidal thoughts ?": "Yes",
    "Work/Study Hours": 10.0,
    "Financial Stress": 5.0,
    "Family History of Mental Illness": "Yes",
}


def test_health_reports_loaded_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "Logistic Regression"


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["depression_risk"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_name"] == "Logistic Regression"


def test_predict_rejects_incomplete_payload(client):
    resp = client.post("/predict", json={"Age": 24})
    assert resp.status_code == 422
