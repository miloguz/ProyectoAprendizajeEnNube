"""Pruebas de las funciones puras de la app de Streamlit.

No se testea el renderizado de widgets (requiere el runtime de Streamlit);
solo la lógica que no depende de él.
"""

from src.app.streamlit_app import load_category_options


def test_load_category_options_returns_known_categories():
    options = load_category_options.__wrapped__()

    assert "Male" not in options["city"]
    assert len(options["city"]) > 0
    assert "Student" in options["profession"]
    assert len(options["degree"]) > 0
