import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.uci_fitness import UciBatchEvaluator, evaluate_config
from src.genetic.engine import GenerationStats, run_genetic_algorithm
from src.genetic.hyperparams_chromosome import (
    ACTIVATION_GENE,
    BATCH_SIZE_GENE,
    DEFAULT_CONFIG,
    DROPOUT_GENE,
    LAYER_COUNT_GENE,
    LEARNING_RATE_GENE,
    WEIGHT_DECAY_GENE,
    WIDTH_GENES,
    decode,
    random_chromosome as random_hyperparameter_chromosome,
)
from src.genetic.mixed_operators import mixed_mutation
from src.genetic.operators import uniform_crossover
from src.genetic.random_search import run_random_search
from src.genetic.selection import tournament_selection
from src.networks.torch_mlp import ACTIVATIONS, MLPConfig
from src.uci.dataset import LetterDataset, load_letter_dataset

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "letter"
POPULATION_SIZE = 20
SEED_OFFSET = 10_000
DEFAULT_SEED = 30_000
WORKERS = 8
METHOD_GENETIC = "Algoritmo genético (cromosoma mixto)"
METHOD_RANDOM = "Búsqueda aleatoria"
METHODS = (METHOD_GENETIC, METHOD_RANDOM)
WIDTH_OPTIONS = (16, 32, 64, 128, 256)
LEARNING_RATE_OPTIONS = (1e-4, 1e-3, 1e-2, 1e-1)
WEIGHT_DECAY_OPTIONS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


@st.cache_resource
def load_dataset_cached(data_directory: str) -> LetterDataset:
    """Loads the letter dataset once per application session."""
    return load_letter_dataset(Path(data_directory))


@st.cache_data
def evaluate_default_config(epochs: int, data_directory: str) -> float:
    """Returns the validation accuracy of the default configuration for a given epoch budget."""
    dataset = load_letter_dataset(Path(data_directory))
    return evaluate_config(DEFAULT_CONFIG, dataset, epochs=epochs, seed=DEFAULT_SEED)


def build_chromosome(
    layer_count: int,
    width_exponents: tuple[int, int, int, int],
    activation_index: int,
    learning_rate_exponent: int,
    weight_decay_exponent: int,
    dropout: float,
    batch_exponent: int,
) -> np.ndarray:
    """Builds the mixed hyperparameter chromosome from the values chosen in the interface."""
    return np.array(
        [
            float(layer_count),
            float(width_exponents[0]),
            float(width_exponents[1]),
            float(width_exponents[2]),
            float(width_exponents[3]),
            float(activation_index),
            float(learning_rate_exponent),
            float(weight_decay_exponent),
            float(dropout),
            float(batch_exponent),
        ],
        dtype=np.float64,
    )


def config_summary(config: MLPConfig) -> dict[str, object]:
    """Summarizes one decoded network configuration with readable Spanish labels."""
    return {
        "Capas ocultas": len(config.hidden_sizes),
        "Anchos": " / ".join(str(size) for size in config.hidden_sizes),
        "Activación": config.activation,
        "Tasa aprendizaje": f"{config.learning_rate:.0e}".replace("e-0", "e-"),
        "Weight decay": f"{config.weight_decay:.0e}".replace("e-0", "e-"),
        "Dropout": f"{config.dropout:.2f}",
        "Tamaño de lote": config.batch_size,
    }


def append_history(origin: str, config: MLPConfig, accuracy: float) -> None:
    """Appends one evaluated configuration to the session history table."""
    history: list[dict[str, object]] = st.session_state.setdefault("letter_history", [])
    history.insert(0, {"Origen": origin, "Exactitud validación": round(accuracy, 4), **config_summary(config)})


def render_manual_configurator(dataset: LetterDataset, epochs: int) -> None:
    """Renders the interactive chromosome builder and trains the chosen configuration."""
    st.markdown(
        "Compón un cromosoma de 10 genes como lo haría el algoritmo genético "
        "y míralo entrenar con descenso por gradiente (el mismo circuito `decode` del proyecto)."
    )
    selection_column, summary_column = st.columns(2, gap="large")
    with selection_column:
        layer_count = st.slider("Número de capas ocultas", 1, 4, 2)
        widths: list[int] = []
        for layer_index in range(layer_count):
            widths.append(
                st.select_slider(
                    f"Ancho de la capa {layer_index + 1}",
                    options=list(WIDTH_OPTIONS),
                    value=WIDTH_OPTIONS[3],
                )
            )
        activation_name = st.selectbox("Función de activación", ACTIVATIONS, index=1)
        learning_rate = st.select_slider(
            "Tasa de aprendizaje",
            options=list(LEARNING_RATE_OPTIONS),
            value=1e-3,
            format_func=lambda value: f"{value:.0e}",
        )
        weight_decay = st.select_slider(
            "Weight decay",
            options=list(WEIGHT_DECAY_OPTIONS),
            value=1e-4,
            format_func=lambda value: f"{value:.0e}",
        )
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
        batch_size = st.select_slider(
            "Tamaño de lote",
            options=list(WIDTH_OPTIONS),
            value=64,
        )
        train_seed = st.number_input("Semilla de entrenamiento", 0, 99999, 1)
        width_exponents = tuple(
            int(round(np.log2(width))) for width in widths + [16] * (4 - len(widths))
        )
        chromosome = build_chromosome(
            layer_count,
            width_exponents,
            ACTIVATIONS.index(activation_name),
            int(round(np.log10(learning_rate))),
            int(round(np.log10(weight_decay))),
            dropout,
            int(round(np.log2(batch_size))),
        )
        config = decode(chromosome)
        if st.button("Entrenar configuración", type="primary"):
            with st.spinner(f"Entrenando {epochs} épocas…"):
                accuracy = evaluate_config(config, dataset, epochs=epochs, seed=int(train_seed))
            append_history("Manual", config, accuracy)
            st.session_state["letter_manual_result"] = {
                "config": config_summary(config),
                "accuracy": accuracy,
            }
    with summary_column:
        st.subheader("Configuración seleccionada")
        st.dataframe(pd.DataFrame([config_summary(decode(chromosome))]), hide_index=True, width="stretch")
        default_accuracy = evaluate_default_config(epochs, str(DATA_DIRECTORY))
        st.caption(f"Referencia por defecto (64-62... red 64-64, ReLU, lr 1e-3): {default_accuracy:.4f} en validación")
        manual_result = st.session_state.get("letter_manual_result")
        if manual_result is not None:
            st.metric(
                "Exactitud en validación",
                f"{manual_result['accuracy']:.4f}",
                delta=f"{manual_result['accuracy'] - default_accuracy:+.4f} vs. por defecto",
            )


def render_search_results(stored: dict[str, object]) -> None:
    """Shows the stored convergence curve and the best configuration found by a search run."""
    best_curve = np.asarray(stored["best_curve"], dtype=np.float64)
    evaluations = np.asarray(stored["evaluations"], dtype=np.int64)
    st.subheader("Resultado de la búsqueda")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Mejor exactitud", f"{best_curve[-1]:.4f}")
    metric_b.metric("Evaluaciones", int(evaluations[-1]))
    metric_c.metric("Épocas por evaluación", int(stored["epochs"]))
    curve_frame = pd.DataFrame({"Mejor hasta la fecha": best_curve}, index=evaluations)
    st.line_chart(
        curve_frame,
        height=260,
        x_label="Evaluaciones",
        y_label="Mejor exactitud en validación",
    )
    st.dataframe(pd.DataFrame([stored["summary"]]), hide_index=True, width="stretch")


def render_search(dataset: LetterDataset) -> None:
    """Renders the automatic search controls and runs the selected method."""
    epochs = st.slider("Épocas de entrenamiento por evaluación", 5, 20, 10, 5)
    st.session_state["letter_epochs"] = epochs
    method = st.selectbox("Método de búsqueda", METHODS)
    budget = st.slider("Presupuesto (evaluaciones)", 40, 240, 60, 20)
    run_seed = st.number_input("Semilla de ejecución", 0, 9999, 0)
    st.caption(
        f"Cada evaluación entrena una red {epochs} épocas en CPU (~unos segundos). "
        "El experimento oficial de la memoria usa 20 épocas y 240 evaluaciones con 5 semillas."
    )
    if st.button("Buscar", type="primary", width="stretch"):
        st.session_state["letter_run_seed"] = int(run_seed)

        def init_fn(rng: np.random.Generator) -> np.ndarray:
            """Samples one random hyperparameter chromosome for the search."""
            return random_hyperparameter_chromosome(rng)

        mutation = functools.partial(mixed_mutation, rate=0.3)
        if method == METHOD_GENETIC:
            progress_bar = st.progress(0.0, text="Evolucionando…")
            live_chart = st.empty()
            generation_values: list[float] = []

            def live_callback(stats: GenerationStats) -> None:
                generation_values.append(stats.best_so_far)
                progress_bar.progress(
                    min(stats.evaluations / budget, 1.0),
                    text=f"Generación {stats.generation + 1}",
                )
                live_chart.line_chart(
                    pd.DataFrame({"Mejor hasta la fecha": generation_values}),
                    height=260,
                )

            with st.spinner("Buscando la mejor configuración…"):
                with UciBatchEvaluator(
                    DATA_DIRECTORY,
                    epochs=epochs,
                    workers=WORKERS,
                    seed=int(run_seed) + SEED_OFFSET,
                ) as evaluator:
                    start = time.perf_counter()
                    result = run_genetic_algorithm(
                        init_fn=init_fn,
                        selection_fn=tournament_selection,
                        crossover_fn=uniform_crossover,
                        mutation_fn=mutation,
                        population_size=POPULATION_SIZE,
                        max_evaluations=budget,
                        population_fitness_fn=evaluator,
                        seed=int(run_seed),
                        on_generation=live_callback,
                    )
                    seconds = time.perf_counter() - start
            progress_bar.progress(1.0, text="Completado")
        else:
            with st.spinner("Muestreando configuraciones aleatorias…"):
                with UciBatchEvaluator(
                    DATA_DIRECTORY,
                    epochs=epochs,
                    workers=WORKERS,
                    seed=int(run_seed) + SEED_OFFSET,
                ) as evaluator:
                    start = time.perf_counter()
                    result = run_random_search(
                        fitness_fn=None,
                        init_fn=init_fn,
                        population_size=POPULATION_SIZE,
                        max_evaluations=budget,
                        seed=int(run_seed),
                        population_fitness_fn=evaluator,
                    )
                    seconds = time.perf_counter() - start
        decoded = decode(result.best_chromosome)
        append_history(method, decoded, float(np.max(np.asarray(result.best_curve))))
        st.session_state["letter_search_result"] = {
            "best_curve": np.asarray(result.best_curve, dtype=np.float64),
            "evaluations": np.asarray(result.evaluations_curve, dtype=np.int64),
            "summary": config_summary(decoded),
            "epochs": epochs,
            "seconds": seconds,
        }
    stored = st.session_state.get("letter_search_result")
    if stored is not None:
        st.caption(f"Tiempo de búsqueda: {stored['seconds']:.1f} s")
        render_search_results(stored)


def render_history() -> None:
    """Shows every configuration evaluated during this application session."""
    history: list[dict[str, object]] = st.session_state.get("letter_history", [])
    if not history:
        return
    st.divider()
    st.subheader("📋 Historial de configuraciones probadas")
    st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")


def render_letter_tab() -> None:
    """Renders the whole Letter Recognition tab with configurator and search."""
    if not DATA_DIRECTORY.exists():
        st.error(
            "No se encontró el dataset preparado. Ejecuta primero "
            "`python -m data.prepare_letter` desde la raíz del repositorio."
        )
        return
    dataset = load_dataset_cached(str(DATA_DIRECTORY))
    st.markdown(
        "La segunda vertiente del proyecto: el algoritmo genético **no toca los pesos** "
        "(los entrena el descenso por gradiente), solo elige la receta: arquitectura e hiperparámetros. "
        "Cada prueba entrena una red en el dataset UCI Letter Recognition y mide su exactitud en validación."
    )
    manual_tab, search_tab = st.tabs(["⚙️ Configurador manual", "🔎 Búsqueda automática"])
    with manual_tab:
        render_manual_configurator(dataset, 10)
    with search_tab:
        render_search(dataset)
    render_history()
