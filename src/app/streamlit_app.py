"""Interfaz Streamlit para estimar el riesgo de depresión de un estudiante.

Formulario que pide las mismas variables que la API (``src/api/schemas.py``)
y llama a ``POST /predict`` del servicio FastAPI — no carga el modelo
directamente, es un cliente de la API (ver README, sección "Interfaz
Streamlit").

Ejecutar con:
    uv run streamlit run src/app/streamlit_app.py

Requiere la API corriendo (local o Docker). Por defecto apunta a
``http://localhost:8000``; se puede sobreescribir con la variable de
entorno ``STREAMLIT_API_URL``.

> ⚠️ Nota ética: esta app estima un **riesgo estadístico** a partir de datos
> autorreportados. **No** es un diagnóstico clínico — cualquier señal de
> alerta real debe derivarse a un profesional de salud mental.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from src.config.constants import RAW_DATASET_PATH
from src.data.loaders import load_raw_data

API_URL = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")

SLEEP_DURATION_OPTIONS = [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours",
    "Others",
]
DIETARY_HABITS_OPTIONS = ["Unhealthy", "Moderate", "Healthy", "Others"]
YES_NO_OPTIONS = ["No", "Yes"]
GENDER_OPTIONS = ["Male", "Female"]


@st.cache_data
def load_category_options() -> dict[str, list[str]]:
    """Carga City/Profession/Degree desde el dataset crudo, para poblar los
    selectboxes con categorías realmente vistas por el modelo en entrenamiento
    (evita que el usuario escriba un valor fuera de dominio)."""
    try:
        df = load_raw_data(RAW_DATASET_PATH)
        return {
            "city": sorted(df["City"].dropna().unique().tolist()),
            "profession": sorted(df["Profession"].dropna().unique().tolist()),
            "degree": sorted(df["Degree"].dropna().unique().tolist()),
        }
    except FileNotFoundError:
        return {"city": [], "profession": [], "degree": []}


def render_form(options: dict[str, list[str]]) -> dict | None:
    """Dibuja el formulario y devuelve el payload listo para /predict, o None
    si aún no se envió."""
    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Demográficos y académicos")
            gender = st.selectbox("Género", GENDER_OPTIONS)
            age = st.number_input("Edad", min_value=10, max_value=80, value=22)
            city = (
                st.selectbox("Ciudad", options["city"])
                if options["city"]
                else st.text_input("Ciudad")
            )
            profession = (
                st.selectbox("Profesión", options["profession"])
                if options["profession"]
                else st.text_input("Profesión", value="Student")
            )
            degree = (
                st.selectbox("Título / carrera", options["degree"])
                if options["degree"]
                else st.text_input("Título / carrera")
            )
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

            st.subheader("Estilo de vida")
            sleep_duration = st.selectbox("Horas de sueño", SLEEP_DURATION_OPTIONS)
            dietary_habits = st.selectbox("Hábitos alimenticios", DIETARY_HABITS_OPTIONS)
            work_study_hours = st.number_input(
                "Horas de estudio/trabajo al día", min_value=0.0, max_value=24.0, value=6.0, step=0.5
            )

        with col2:
            st.subheader("Presión y satisfacción (0-5)")
            academic_pressure = st.slider("Presión académica", 0.0, 5.0, 3.0, 0.5)
            work_pressure = st.slider("Presión laboral", 0.0, 5.0, 0.0, 0.5)
            study_satisfaction = st.slider("Satisfacción con los estudios", 0.0, 5.0, 3.0, 0.5)
            job_satisfaction = st.slider("Satisfacción laboral", 0.0, 5.0, 0.0, 0.5)
            financial_stress = st.slider("Estrés financiero", 0.0, 5.0, 3.0, 0.5)

            st.subheader("Salud mental")
            suicidal_thoughts = st.selectbox(
                "¿Ha tenido pensamientos suicidas alguna vez?", YES_NO_OPTIONS
            )
            family_history = st.selectbox(
                "¿Antecedentes familiares de enfermedad mental?", YES_NO_OPTIONS
            )

        submitted = st.form_submit_button("Evaluar riesgo")

    if not submitted:
        return None

    return {
        "Gender": gender,
        "Age": age,
        "City": city,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
    }


def call_predict(payload: dict) -> dict:
    """Llama a POST /predict y devuelve la respuesta ya parseada."""
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def render_result(result: dict) -> None:
    risk = result["depression_risk"]
    probability = result["probability"]
    threshold = result["threshold"]

    if risk == 1:
        st.error(
            f"⚠️ Riesgo detectado — probabilidad estimada: **{probability:.1%}** "
            f"(umbral aplicado: {threshold:.2f})"
        )
        st.write(
            "Se recomienda buscar apoyo de un profesional de salud mental. Esta "
            "herramienta **no diagnostica**: es una estimación estadística de riesgo."
        )
    else:
        st.success(
            f"✅ Sin riesgo detectado — probabilidad estimada: **{probability:.1%}** "
            f"(umbral aplicado: {threshold:.2f})"
        )
        st.write(
            "Aunque el modelo no detectó riesgo, si sientes que necesitas apoyo, "
            "buscar ayuda profesional siempre es una buena decisión."
        )

    st.caption(f"Modelo: {result['model_name']}")


def main() -> None:
    st.set_page_config(page_title="Riesgo de depresión en estudiantes", page_icon="🧠")
    st.title("🧠 Riesgo de depresión en estudiantes")
    st.caption(
        "Herramienta de análisis estadístico con fines académicos. **No** es un "
        "diagnóstico clínico ni sustituye la valoración de un profesional de salud mental."
    )

    with st.sidebar:
        st.header("Conexión")
        st.write(f"API: `{API_URL}`")
        try:
            health = requests.get(f"{API_URL}/health", timeout=3).json()
            st.success(f"Modelo activo: {health.get('model_name')} (v{health.get('version')})")
        except requests.RequestException:
            st.error(
                "No se pudo conectar a la API. Levántala con "
                "`uv run uvicorn src.api.main:app --reload` o con Docker."
            )

    options = load_category_options()
    payload = render_form(options)

    if payload is None:
        return

    try:
        result = call_predict(payload)
    except requests.RequestException as exc:
        st.error(f"Error al llamar a la API: {exc}")
        return

    render_result(result)


if __name__ == "__main__":
    main()
