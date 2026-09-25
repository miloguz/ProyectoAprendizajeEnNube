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

import pandas as pd
import requests
import streamlit as st

from src.config.constants import RAW_DATASET_PATH
from src.data.loaders import load_raw_data

API_URL = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")

# El modelo se entrenó con las categorías originales del dataset (en inglés);
# la UI se muestra en español traduciendo solo la ETIQUETA visible
# (st.selectbox(..., format_func=...)) — el valor real que viaja a la API no
# cambia, para no generar categorías fuera de dominio para el OneHotEncoder.
GENDER_OPTIONS = ["Male", "Female"]
GENDER_LABELS = {"Male": "Masculino", "Female": "Femenino"}

SLEEP_DURATION_OPTIONS = [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours",
    "Others",
]
SLEEP_DURATION_LABELS = {
    "Less than 5 hours": "Menos de 5 horas",
    "5-6 hours": "5-6 horas",
    "7-8 hours": "7-8 horas",
    "More than 8 hours": "Más de 8 horas",
    "Others": "Otro",
}

DIETARY_HABITS_OPTIONS = ["Unhealthy", "Moderate", "Healthy", "Others"]
DIETARY_HABITS_LABELS = {
    "Unhealthy": "Poco saludables",
    "Moderate": "Moderados",
    "Healthy": "Saludables",
    "Others": "Otros",
}

YES_NO_OPTIONS = ["No", "Yes"]
YES_NO_LABELS = {"No": "No", "Yes": "Sí"}

PROFESSION_LABELS = {
    "Architect": "Arquitecto/a",
    "Chef": "Chef",
    "Civil Engineer": "Ingeniero/a Civil",
    "Content Writer": "Redactor/a de Contenidos",
    "Digital Marketer": "Especialista en Marketing Digital",
    "Doctor": "Médico/a",
    "Educational Consultant": "Consultor/a Educativo/a",
    "Entrepreneur": "Emprendedor/a",
    "Lawyer": "Abogado/a",
    "Manager": "Gerente",
    "Pharmacist": "Farmacéutico/a",
    "Student": "Estudiante",
    "Teacher": "Profesor/a",
    "UX/UI Designer": "Diseñador/a UX/UI",
}

# Título completo en español + sigla original entre paréntesis (la sigla es
# la que viaja a la API/modelo).
DEGREE_LABELS = {
    "B.Arch": "Arquitectura (B.Arch)",
    "B.Com": "Comercio (B.Com)",
    "B.Ed": "Educación (B.Ed)",
    "B.Pharm": "Farmacia (B.Pharm)",
    "B.Tech": "Ingeniería / Tecnología (B.Tech)",
    "BA": "Artes y Humanidades (BA)",
    "BBA": "Administración de Empresas (BBA)",
    "BCA": "Aplicaciones de Computación (BCA)",
    "BE": "Ingeniería (BE)",
    "BHM": "Gestión Hotelera (BHM)",
    "BSc": "Ciencias (BSc)",
    "Class 12": "Bachillerato (grado 12)",
    "LLB": "Derecho (LLB)",
    "LLM": "Maestría en Derecho (LLM)",
    "M.Com": "Maestría en Comercio (M.Com)",
    "M.Ed": "Maestría en Educación (M.Ed)",
    "M.Pharm": "Maestría en Farmacia (M.Pharm)",
    "M.Tech": "Maestría en Ingeniería / Tecnología (M.Tech)",
    "MA": "Maestría en Artes y Humanidades (MA)",
    "MBA": "Maestría en Administración de Empresas (MBA)",
    "MBBS": "Medicina (MBBS)",
    "MCA": "Maestría en Aplicaciones de Computación (MCA)",
    "MD": "Doctorado en Medicina (MD)",
    "ME": "Maestría en Ingeniería (ME)",
    "MHM": "Maestría en Gestión Hotelera (MHM)",
    "MSc": "Maestría en Ciencias (MSc)",
    "Others": "Otro",
    "PhD": "Doctorado (PhD)",
}

MODEL_NAME_LABELS = {
    "Logistic Regression": "Regresión Logística",
    "Decision Tree": "Árbol de Decisión",
    "Random Forest": "Bosque Aleatorio (Random Forest)",
}


def _model_label(name: str | None) -> str:
    return MODEL_NAME_LABELS.get(name, name or "desconocido")


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
            gender = st.selectbox(
                "Género", GENDER_OPTIONS, format_func=lambda v: GENDER_LABELS.get(v, v)
            )
            age = st.number_input("Edad", min_value=10, max_value=80, value=22)
            city = (
                st.selectbox("Ciudad", options["city"])
                if options["city"]
                else st.text_input("Ciudad")
            )
            profession = (
                st.selectbox(
                    "Profesión",
                    options["profession"],
                    format_func=lambda v: PROFESSION_LABELS.get(v, v),
                )
                if options["profession"]
                else st.text_input("Profesión", value="Student")
            )
            degree = (
                st.selectbox(
                    "Título / carrera",
                    options["degree"],
                    format_func=lambda v: DEGREE_LABELS.get(v, v),
                )
                if options["degree"]
                else st.text_input("Título / carrera")
            )
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

            st.subheader("Estilo de vida")
            sleep_duration = st.selectbox(
                "Horas de sueño",
                SLEEP_DURATION_OPTIONS,
                format_func=lambda v: SLEEP_DURATION_LABELS.get(v, v),
            )
            dietary_habits = st.selectbox(
                "Hábitos alimenticios",
                DIETARY_HABITS_OPTIONS,
                format_func=lambda v: DIETARY_HABITS_LABELS.get(v, v),
            )
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
                "¿Ha tenido pensamientos suicidas alguna vez?",
                YES_NO_OPTIONS,
                format_func=lambda v: YES_NO_LABELS.get(v, v),
            )
            family_history = st.selectbox(
                "¿Antecedentes familiares de enfermedad mental?",
                YES_NO_OPTIONS,
                format_func=lambda v: YES_NO_LABELS.get(v, v),
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


def fetch_models() -> list[dict]:
    """Llama a GET /models y devuelve la lista de modelos disponibles."""
    response = requests.get(f"{API_URL}/models", timeout=5)
    response.raise_for_status()
    return response.json()["models"]


def call_predict(payload: dict, model_name: str | None = None) -> dict:
    """Llama a POST /predict con el modelo elegido y devuelve la respuesta ya parseada."""
    params = {"model_name": model_name} if model_name else None
    response = requests.post(f"{API_URL}/predict", json=payload, params=params, timeout=10)
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

    st.caption(f"Modelo: {_model_label(result['model_name'])}")


def render_prediction_tab(models_info: list[dict]) -> None:
    """Formulario de predicción, con selector del modelo a usar."""
    model_names = [m["name"] for m in models_info]
    candidate_name = next((m["name"] for m in models_info if m["is_candidate"]), None)
    default_index = model_names.index(candidate_name) if candidate_name in model_names else 0

    selected_model = st.selectbox(
        "Modelo a usar para la predicción",
        model_names,
        index=default_index,
        format_func=_model_label,
        help=(
            "Por defecto, el modelo candidato (mayor F1 en test, alias "
            "MasterModel en el MLflow Model Registry). Puedes comparar con "
            "los otros dos."
        ),
    )

    options = load_category_options()
    payload = render_form(options)

    if payload is None:
        return

    try:
        result = call_predict(payload, selected_model)
    except requests.RequestException as exc:
        st.error(f"Error al llamar a la API: {exc}")
        return

    render_result(result)


def render_dashboard_tab(models_info: list[dict]) -> None:
    """Comparativa de los 3 modelos: métricas, umbral, nº de experimentos e
    hiperparámetros — pensado para mostrar en la exposición."""
    st.subheader("Comparación de modelos")

    rows = [
        {
            "Modelo": _model_label(m["name"]) + (" 🏆" if m["is_candidate"] else ""),
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "Recall": m["recall_pos"],
            "Umbral": m["threshold"],
            "Nº experimentos (MLflow)": m["n_experiments"],
        }
        for m in models_info
    ]
    df = pd.DataFrame(rows).set_index("Modelo")
    st.dataframe(
        df.style.format(
            {"F1": "{:.3f}", "ROC-AUC": "{:.3f}", "Recall": "{:.3f}", "Umbral": "{:.2f}"}
        ),
        width="stretch",
    )
    st.caption("🏆 = modelo candidato (el que sirve `/predict` por defecto).")

    st.bar_chart(df[["F1", "ROC-AUC", "Recall"]])

    st.subheader("Hiperparámetros por modelo (tras el tuning)")
    for m in models_info:
        label = _model_label(m["name"]) + (" — candidato" if m["is_candidate"] else "")
        with st.expander(label):
            st.json(m["best_params"])


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
            st.success(
                f"Modelo candidato: {_model_label(health.get('model_name'))} "
                f"(v{health.get('version')})"
            )
            st.caption(f"{health.get('n_models', 0)} modelos disponibles")
        except requests.RequestException:
            st.error(
                "No se pudo conectar a la API. Levántala con "
                "`uv run uvicorn src.api.main:app --reload` o con Docker."
            )

    try:
        models_info = fetch_models()
    except requests.RequestException:
        models_info = []

    tab_predict, tab_dashboard = st.tabs(["🔮 Predicción", "📊 Dashboard"])

    with tab_predict:
        if not models_info:
            st.warning(
                "No se pudo obtener la lista de modelos desde la API. Verifica que "
                "esté corriendo y que el pipeline de orquestación se haya ejecutado."
            )
        else:
            render_prediction_tab(models_info)

    with tab_dashboard:
        if not models_info:
            st.info("Sin datos de modelos para mostrar.")
        else:
            render_dashboard_tab(models_info)


if __name__ == "__main__":
    main()
