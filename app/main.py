import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import streamlit as st

from app.cartpole_view import render_cartpole_tab
from app.letter_view import render_letter_tab


def main() -> None:
    """Renderiza la interfaz de demostración con sus dos pestañas."""
    st.set_page_config(page_title="Neuroevolución · Demo", page_icon="🧬", layout="wide")
    st.title("Neuroevolución: optimización de redes neuronales con algoritmos genéticos")
    st.caption(
        "Herramienta de demostración para la presentación · "
        "Optimización y Heurística (ULPGC) · Propuesta A4"
    )
    cartpole_tab, letter_tab = st.tabs(
        [
            "CartPole-v1 · control sin gradientes",
            "Letter Recognition · arquitectura e hiperparámetros (UCI)",
        ]
    )
    with cartpole_tab:
        render_cartpole_tab()
    with letter_tab:
        render_letter_tab()


if __name__ == "__main__":
    main()
