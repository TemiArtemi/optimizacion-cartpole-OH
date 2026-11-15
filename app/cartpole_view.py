import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.baselines.reinforce import run_reinforce
from src.environments.cartpole_eval import (
    CARTPOLE_ARCHITECTURE,
    CartpoleBatchEvaluator,
    evaluate_chromosome,
    play_episode,
)
from src.genetic.engine import EvolutionResult, GenerationStats, run_genetic_algorithm
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.genetic.random_search import run_random_search
from src.genetic.selection import tournament_selection
from src.networks.numpy_mlp import NumpyMLP, random_chromosome
from src.utils.metrics import evaluations_to_target

METHOD_GENETIC = "Algoritmo genético (torneo k=3)"
METHOD_RANDOM = "Búsqueda aleatoria"
METHOD_REINFORCE = "REINFORCE (tasa 0.001)"
METHODS = (METHOD_GENETIC, METHOD_RANDOM, METHOD_REINFORCE)
POPULATION_SIZE = 50
EPISODES_PER_EVALUATION = 3
TARGET_FITNESS = 500.0
SEED_OFFSET = 10_000
WORKERS = 4
CONTROLLER_DIRECTORY = Path(__file__).resolve().parent / "controllers"


def init_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Samples one random network chromosome for the CartPole architecture."""
    return random_chromosome(CARTPOLE_ARCHITECTURE, rng)


def make_fitness(run_seed: int) -> object:
    """Builds a reproducible fitness that scores evaluations on fresh initial states."""
    evaluation_rng = np.random.default_rng(run_seed + SEED_OFFSET)

    def fitness(chromosome: np.ndarray) -> float:
        """Evaluates one chromosome with newly drawn initial conditions."""
        seed = int(evaluation_rng.integers(1_000_000))
        return evaluate_chromosome(chromosome, episodes=EPISODES_PER_EVALUATION, seed=seed)

    return fitness


def train_genetic(budget: int, run_seed: int, on_generation: object) -> tuple[EvolutionResult, float]:
    """Runs the genetic algorithm on a parallel evaluator and returns it with the elapsed seconds."""
    mutation = functools.partial(gaussian_mutation, rate=0.1, sigma=0.1)
    with CartpoleBatchEvaluator(
        architecture=CARTPOLE_ARCHITECTURE,
        episodes=EPISODES_PER_EVALUATION,
        workers=WORKERS,
        seed=run_seed + SEED_OFFSET,
    ) as evaluator:
        start = time.perf_counter()
        result = run_genetic_algorithm(
            init_fn=init_chromosome,
            selection_fn=tournament_selection,
            crossover_fn=one_point_crossover,
            mutation_fn=mutation,
            population_size=POPULATION_SIZE,
            max_evaluations=budget,
            population_fitness_fn=evaluator,
            seed=run_seed,
            on_generation=on_generation,
        )
        return result, time.perf_counter() - start


def train_random(budget: int, run_seed: int) -> tuple[EvolutionResult, float]:
    """Runs random search with the same budget and returns it with the elapsed seconds."""
    start = time.perf_counter()
    result = run_random_search(
        fitness_fn=make_fitness(run_seed),
        init_fn=init_chromosome,
        population_size=POPULATION_SIZE,
        max_evaluations=budget,
        seed=run_seed,
    )
    return result, time.perf_counter() - start


def train_reinforce(budget: int, run_seed: int) -> tuple[object, float]:
    """Trains REINFORCE for the equivalent episode budget and returns it with the elapsed seconds."""
    start = time.perf_counter()
    result = run_reinforce(
        CARTPOLE_ARCHITECTURE,
        episodes=budget * EPISODES_PER_EVALUATION,
        seed=run_seed,
    )
    return result, time.perf_counter() - start


def build_animation(states: np.ndarray, title: str) -> go.Figure:
    """Builds a plotly animation of the cart and pole from one recorded state trajectory."""
    frames: list[go.Frame] = []
    for step, state in enumerate(states):
        position = float(state[0])
        angle = float(state[2])
        pivot_x, pivot_y = position, 0.15
        top_x = pivot_x + 0.5 * np.sin(angle)
        top_y = pivot_y + 0.5 * np.cos(angle)
        frames.append(
            go.Frame(
                name=str(step),
                data=[
                    go.Scatter(
                        x=[pivot_x, pivot_x],
                        y=[pivot_y, pivot_y + 0.75],
                        mode="lines",
                        line=dict(color="gray", width=2, dash="dash"),
                    ),
                    go.Scatter(
                        x=[pivot_x],
                        y=[0.08],
                        mode="markers",
                        marker=dict(size=26, color="royalblue", symbol="square"),
                    ),
                    go.Scatter(
                        x=[pivot_x, top_x],
                        y=[pivot_y, top_y],
                        mode="lines",
                        line=dict(color="darkorange", width=6),
                    ),
                ],
            )
        )
    figure = go.Figure(data=frames[0].data, frames=frames)
    figure.update_layout(
        title=title,
        shapes=[
            dict(
                type="line",
                x0=-2.6,
                x1=2.6,
                y0=0.0,
                y1=0.0,
                line=dict(color="gray", width=3),
            )
        ],
        xaxis=dict(range=[-2.6, 2.6], title="posición del carrito (m)"),
        yaxis=dict(range=[-0.4, 1.5], title="altura (m)", scaleanchor="x"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Reproducir",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 25, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    )
                ],
            )
        ],
        height=430,
        showlegend=False,
    )
    return figure


def play_controller_episode(chromosome: np.ndarray, episode_seed: int) -> tuple[np.ndarray, int]:
    """Plays one episode with the given controller and returns its states and score."""
    network = NumpyMLP(CARTPOLE_ARCHITECTURE, chromosome)
    trace = play_episode(network.forward, seed=episode_seed)
    return trace.states, len(trace.actions)


def save_controller(chromosome: np.ndarray, method: str, run_seed: int, fitness: float) -> Path:
    """Stores the best controller of a run so it can be replayed instantly later."""
    CONTROLLER_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CONTROLLER_DIRECTORY / f"cartpole_seed{run_seed}.npz"
    np.savez(
        path,
        chromosome=chromosome,
        architecture=np.array(CARTPOLE_ARCHITECTURE),
        method=np.str_(method),
        fitness=np.float64(fitness),
    )
    return path


def list_controllers() -> list[str]:
    """Returns the file names of the saved controllers, sorted for stable selection."""
    if not CONTROLLER_DIRECTORY.exists():
        return []
    return sorted(path.name for path in CONTROLLER_DIRECTORY.glob("*.npz"))


def load_controller(file_name: str) -> tuple[np.ndarray, dict[str, object]]:
    """Reads one saved controller file back into memory with its metadata."""
    with np.load(CONTROLLER_DIRECTORY / file_name) as archive:
        chromosome = np.array(archive["chromosome"], dtype=np.float64)
        metadata = {
            "method": str(archive["method"]),
            "fitness": float(archive["fitness"]),
        }
    return chromosome, metadata


def render_training_results(result: object, method: str, seconds: float) -> None:
    """Stores the training outcome in the session and shows metrics plus its convergence curve."""
    if method == METHOD_REINFORCE:
        best_curve = np.asarray(result.best_curve, dtype=np.float64)
        evaluations = np.arange(1, len(best_curve) + 1, dtype=np.int64)
        best_fitness = float(np.max(best_curve))
        best_chromosome = np.asarray(result.final_chromosome, dtype=np.float64)
    else:
        best_curve = np.asarray(result.best_curve, dtype=np.float64)
        evaluations = np.asarray(result.evaluations_curve, dtype=np.int64)
        best_fitness = float(result.best_fitness)
        best_chromosome = np.asarray(result.best_chromosome, dtype=np.float64)
    st.session_state["cartpole_result"] = {
        "method": method,
        "best_curve": best_curve,
        "evaluations": evaluations,
        "best_fitness": best_fitness,
        "best_chromosome": best_chromosome,
        "seconds": seconds,
    }


def show_training_results() -> None:
    """Renders the stored training outcome: metrics, convergence curve and save button."""
    stored = st.session_state.get("cartpole_result")
    if stored is None:
        return
    best_fitness = stored["best_fitness"]
    evaluations = stored["evaluations"]
    best_curve = stored["best_curve"]
    target = evaluations_to_target(best_curve, evaluations, TARGET_FITNESS)
    st.subheader(f"Resultado · {stored['method']}")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Mejor fitness", f"{best_fitness:.1f}")
    metric_b.metric("Evaluaciones hasta 500", "—" if target is None else f"{target:,}".replace(",", "."))
    metric_c.metric("Tiempo de búsqueda", f"{stored['seconds']:.1f} s")
    curve_frame = pd.DataFrame({"Mejor hasta la fecha": best_curve}, index=evaluations)
    axis_label = "Episodios jugados" if stored["method"] == METHOD_REINFORCE else "Evaluaciones"
    st.line_chart(curve_frame, height=260, x_label=axis_label, y_label="Mejor fitness hasta la fecha")
    st.caption(
        "La curva muestra el mejor valor visto hasta cada punto: nunca baja (gracias al elitismo) "
        "y se queda plana al alcanzar 500, la puntuación máxima de CartPole (500 pasos × 1 punto). "
        "Con esta red, el algoritmo genético resuelve la tarea en torno a 300–500 evaluaciones "
        "—la memoria reporta 470 de media—, por lo que una curva que sube rápido y se estanca es el comportamiento esperado."
    )
    if st.button("💾 Guardar mejor controlador"):
        path = save_controller(
            stored["best_chromosome"],
            stored["method"],
            int(st.session_state.get("cartpole_run_seed", 0)),
            best_fitness,
        )
        st.success(f"Controlador guardado en {path.name}")


def render_animation_section() -> None:
    """Shows the animation controls for the current or any saved controller."""
    st.divider()
    st.subheader("🎬 Ver el controlador en acción")
    st.caption(
        "El episodio termina en cuanto el palo se inclina más de 12° o el carrito sale de la vía "
        "(reglas del juego), por lo que nunca verás una caída completa: la línea discontinua gris "
        "es la vertical de referencia para apreciar la inclinación del palo."
    )
    stored = st.session_state.get("cartpole_result")
    animation_column, saved_column = st.columns(2)
    with animation_column:
        episode_seed = st.number_input("Semilla del episodio", 0, 99999, 42, key="cartpole_episode_seed")
        if stored is not None and st.button("Reproducir el controlador entrenado"):
            states, score = play_controller_episode(stored["best_chromosome"], int(episode_seed))
            figure = build_animation(
                states,
                f"Controlador entrenado · pasos sobrevividos: {score}",
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        elif stored is None:
            st.caption("Entrena un controlador o carga uno guardado para verlo en movimiento.")
    with saved_column:
        controllers = list_controllers()
        if not controllers:
            st.caption("Aún no hay controladores guardados.")
            return
        selected = st.selectbox("Controladores guardados", controllers)
        if st.button("Reproducir controlador guardado"):
            chromosome, metadata = load_controller(selected)
            states, score = play_controller_episode(chromosome, int(episode_seed))
            figure = build_animation(
                states,
                f"{metadata['method']} · guardado con fitness {metadata['fitness']:.1f} · pasos: {score}",
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render_cartpole_tab() -> None:
    """Renders the whole CartPole tab: controls, training, results and animation."""
    st.markdown(
        "Entrena un controlador de la red $(4,8,2)$ sin gradientes y míralo mantener "
        "el polo en equilibrio. Todos los métodos comparten el mismo presupuesto de "
        f"evaluaciones (cada evaluación = {EPISODES_PER_EVALUATION} episodios)."
    )
    controls_column, results_column = st.columns([1, 2], gap="large")
    with controls_column:
        method = st.selectbox("Método de búsqueda", METHODS, key="cartpole_method")
        budget = st.slider("Presupuesto (evaluaciones)", 500, 3000, 1000, 250, key="cartpole_budget")
        run_seed = st.number_input("Semilla de ejecución", 0, 9999, 0, key="cartpole_search_seed")
        if method == METHOD_REINFORCE:
            st.caption(f"REINFORCE entrena {budget * EPISODES_PER_EVALUATION:,} episodios con esta misma semilla.".replace(",", "."))
        else:
            st.caption(f"Se evaluarán {budget:,} candidatos con {EPISODES_PER_EVALUATION} episodios cada uno.".replace(",", "."))
        train_clicked = st.button("Entrenar", type="primary", width="stretch")
    with results_column:
        if train_clicked:
            st.session_state["cartpole_run_seed"] = int(run_seed)
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

                with st.spinner("Evolucionando…"):
                    result, seconds = train_genetic(int(budget), int(run_seed), live_callback)
                progress_bar.progress(1.0, text="Completado")
            elif method == METHOD_RANDOM:
                with st.spinner("Muestreando candidatos aleatorios…"):
                    result, seconds = train_random(int(budget), int(run_seed))
            else:
                with st.spinner("Entrenando la política con REINFORCE…"):
                    result, seconds = train_reinforce(int(budget), int(run_seed))
            render_training_results(result, method, float(seconds))
        show_training_results()
    render_animation_section()
