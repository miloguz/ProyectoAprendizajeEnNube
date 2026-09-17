"""Constantes y rutas centrales del proyecto."""

from pathlib import Path

# --- Rutas del proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Dataset ---
RAW_DATASET_PATH = RAW_DATA_DIR / "student_depression.csv"

# --- Columna objetivo ---
TARGET = "Depression"

# --- Columnas numéricas ---
NUMERIC_FEATURES = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
]

# --- Columnas categóricas ---
CATEGORICAL_FEATURES = [
    "Gender",
    "City",
    "Profession",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]

# --- Semilla para reproducibilidad ---
RANDOM_STATE = 42
