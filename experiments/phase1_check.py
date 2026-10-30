import functools
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.environments.cartpole_eval import evaluate_chromosome
from src.genetic.engine import EvolutionResult, run_genetic_algorithm
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.genetic.random_search import run_random_search
from src.genetic.selection import tournament_selection
from src.networks.numpy_mlp import random_chromosome

EXPERIMENT_ARCHITECTURE: tuple[int, ...] = (4, 16, 16, 2)
POPULATION_SIZE = 50
MAX_EVALUATIONS = 3000
EPISODES_PER_EVALUATION = 3
TARGET_FITNESS = 500.0
SEED_OFFSET = 10_000
RUN_SEEDS = (0, 1, 2, 3, 4)
FIGURE_PATH = Path(__file__).resolve().parents[1] / "results" / "figures" / "phase1_ga_vs_random.png"


def init_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Samples one random network chromosome for the experiment architecture."""
    return random_chromosome(EXPERIMENT_ARCHITECTURE, rng)


def make_fitness(run_seed: int) -> object:
    """Builds a reproducible fitness that scores every evaluation on fresh initial states."""
    evaluation_rng = np.random.default_rng(run_seed + SEED_OFFSET)

    def fitness(chromosome: np.ndarray) -> float:
        """Evaluates one chromosome on CartPole with newly drawn initial conditions."""
        seed = int(evaluation_rng.integers(1_000_000))
        return evaluate_chromosome(
            chromosome,
            architecture=EXPERIMENT_ARCHITECTURE,
            episodes=EPISODES_PER_EVALUATION,
            seed=seed,
        )

    return fitness


def history_matrix(results: list[EvolutionResult]) -> np.ndarray:
    """Stacks the best-so-far curves of several runs into one matrix."""
    return np.stack([np.array([entry.best_so_far for entry in result.history]) for result in results])


def evaluations_to_target(result: EvolutionResult) -> int | None:
    """Returns the evaluation count at which the target was first reached, if ever."""
    for entry in result.history:
        if entry.best_so_far >= TARGET_FITNESS:
            return entry.evaluations
    return None


def run_one_seed(seed: int) -> tuple[EvolutionResult, EvolutionResult, float, float]:
    """Runs the genetic algorithm and random search for one seed and times both."""
    fitness = make_fitness(seed)
    mutation = functools.partial(gaussian_mutation, rate=0.1, sigma=0.1)
    start = time.perf_counter()
    genetic = run_genetic_algorithm(
        fitness_fn=fitness,
        init_fn=init_chromosome,
        selection_fn=tournament_selection,
        crossover_fn=one_point_crossover,
        mutation_fn=mutation,
        population_size=POPULATION_SIZE,
        max_evaluations=MAX_EVALUATIONS,
        seed=seed,
    )
    genetic_seconds = time.perf_counter() - start
    start = time.perf_counter()
    random_baseline = run_random_search(
        fitness_fn=fitness,
        init_fn=init_chromosome,
        population_size=POPULATION_SIZE,
        max_evaluations=MAX_EVALUATIONS,
        seed=seed,
    )
    random_seconds = time.perf_counter() - start
    return genetic, random_baseline, genetic_seconds, random_seconds


def plot_convergence(genetic_curves: np.ndarray, random_curves: np.ndarray) -> None:
    """Saves the mean convergence curve of both methods with one standard deviation bands."""
    evaluations = np.arange(1, genetic_curves.shape[1] + 1) * POPULATION_SIZE
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for curves, label, color in (
        (genetic_curves, "Algoritmo genético (torneo k=3)", "tab:blue"),
        (random_curves, "Búsqueda aleatoria", "tab:orange"),
    ):
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        axis.plot(evaluations, mean_curve, label=label, color=color)
        axis.fill_between(evaluations, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)
    axis.set_xlabel("Número de evaluaciones")
    axis.set_ylabel("Recompensa media (mejor hasta la fecha)")
    axis.set_title("CartPole-v1 (red 4-16-16-2): algoritmo genético frente a búsqueda aleatoria")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    """Compares the genetic algorithm against random search on CartPole and saves the figure."""
    genetic_results: list[EvolutionResult] = []
    random_results: list[EvolutionResult] = []
    genetic_times: list[float] = []
    random_times: list[float] = []
    for seed in RUN_SEEDS:
        genetic, random_baseline, genetic_seconds, random_seconds = run_one_seed(seed)
        genetic_results.append(genetic)
        random_results.append(random_baseline)
        genetic_times.append(genetic_seconds)
        random_times.append(random_seconds)
        print(
            f"seed {seed}: genetic={genetic.best_fitness:.1f} at {evaluations_to_target(genetic)} "
            f"({genetic_seconds:.1f}s) random={random_baseline.best_fitness:.1f} "
            f"at {evaluations_to_target(random_baseline)} ({random_seconds:.1f}s)"
        )
    genetic_finals = np.array([result.best_fitness for result in genetic_results])
    random_finals = np.array([result.best_fitness for result in random_results])
    genetic_targets = [evaluations_to_target(result) for result in genetic_results]
    random_targets = [evaluations_to_target(result) for result in random_results]
    print(
        f"genetic mean={genetic_finals.mean():.1f} std={genetic_finals.std(ddof=1):.1f} "
        f"solved={sum(value is not None for value in genetic_targets)}/{len(RUN_SEEDS)} "
        f"| random mean={random_finals.mean():.1f} std={random_finals.std(ddof=1):.1f} "
        f"solved={sum(value is not None for value in random_targets)}/{len(RUN_SEEDS)} "
        f"| budget={MAX_EVALUATIONS} evaluations per run"
    )
    print(f"genetic evaluations to reach {TARGET_FITNESS:.0f}: {genetic_targets}")
    print(f"random evaluations to reach {TARGET_FITNESS:.0f}: {random_targets}")
    plot_convergence(history_matrix(genetic_results), history_matrix(random_results))
    print(f"figure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
