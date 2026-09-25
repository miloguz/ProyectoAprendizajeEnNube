"""Pruebas de la API de predicción (FastAPI).

Entrena un pipeline pequeño por cada uno de los 3 modelos sobre una muestra
del dataset y los guarda como los artefactos que la API espera (mismo
formato que produce ``src/orchestration/flow.py``), para no depender de
haber corrido el pipeline de orquestación completo antes de testear.
"""

import os

import joblib
import pytest
from fastapi.testclient import TestClient

from src.data.loaders import load_raw_data
from src.features.engineering import split_X_y
from src.models.train import build_pipeline, get_models

CANDIDATE_MODEL = "Logistic Regression"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    df = load_raw_data()
    X, y = split_X_y(df)
    sample = X.sample(n=200, random_state=0)
    sample_y = y.loc[sample.index]

    models_dir = tmp_path_factory.mktemp("models")
    for i, (name, model) in enumerate(get_models().items()):
        pipeline = build_pipeline(model)
        pipeline.fit(sample, sample_y)
        joblib.dump(
            {
                "pipeline": pipeline,
                "model_name": name,
                "threshold": 0.5,
                "best_params": {"dummy_param": i},
                "metrics": {"f1": 0.80 + i * 0.01, "roc_auc": 0.9, "recall_pos": 0.85},
                "n_experiments": i + 1,
                "is_candidate": name == CANDIDATE_MODEL,
            },
            models_dir / f"model_{i}.joblib",
        )

    previous = os.environ.get("MODELS_DIR")
    os.environ["MODELS_DIR"] = str(models_dir)

    from src.api.main import app as fastapi_app

    yield fastapi_app

    if previous is None:
        os.environ.pop("MODELS_DIR", None)
    else:
        os.environ["MODELS_DIR"] = previous


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


def test_health_reports_candidate_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == CANDIDATE_MODEL
    assert body["n_models"] == 3


def test_list_models_returns_all_three(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate_model"] == CANDIDATE_MODEL
    names = {m["name"] for m in body["models"]}
    assert names == {"Logistic Regression", "Decision Tree", "Random Forest"}
    candidate_flags = {m["name"]: m["is_candidate"] for m in body["models"]}
    assert candidate_flags[CANDIDATE_MODEL] is True
    assert sum(candidate_flags.values()) == 1


def test_predict_uses_candidate_by_default(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["depression_risk"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_name"] == CANDIDATE_MODEL


def test_predict_can_select_another_model(client):
    resp = client.post("/predict", json=VALID_PAYLOAD, params={"model_name": "Decision Tree"})
    assert resp.status_code == 200
    assert resp.json()["model_name"] == "Decision Tree"


def test_predict_rejects_unknown_model(client):
    resp = client.post("/predict", json=VALID_PAYLOAD, params={"model_name": "No Existe"})
    assert resp.status_code == 404


def test_predict_rejects_incomplete_payload(client):
    resp = client.post("/predict", json={"Age": 24})
    assert resp.status_code == 422
