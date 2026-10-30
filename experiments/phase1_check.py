import functools
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.environments.cartpole_eval import evaluate_chromosome
from src.genetic.engine import EvolutionResult, run_genetic_algorithm
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.genetic.random_search import run_random_search
from src.genetic.selection import tournament_selection
from src.networks.numpy_mlp import random_chromosome
from src.utils.metrics import evaluations_to_target, mean_std, success_rate
from src.utils.plotting import plot_convergence

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
        genetic_target = evaluations_to_target(genetic.best_curve, genetic.evaluations_curve, TARGET_FITNESS)
        random_target = evaluations_to_target(
            random_baseline.best_curve, random_baseline.evaluations_curve, TARGET_FITNESS
        )
        print(
            f"seed {seed}: genetic={genetic.best_fitness:.1f} at {genetic_target} "
            f"({genetic_seconds:.1f}s) random={random_baseline.best_fitness:.1f} "
            f"at {random_target} ({random_seconds:.1f}s)"
        )
    genetic_finals = np.array([result.best_fitness for result in genetic_results])
    random_finals = np.array([result.best_fitness for result in random_results])
    genetic_targets = [
        evaluations_to_target(result.best_curve, result.evaluations_curve, TARGET_FITNESS)
        for result in genetic_results
    ]
    random_targets = [
        evaluations_to_target(result.best_curve, result.evaluations_curve, TARGET_FITNESS)
        for result in random_results
    ]
    genetic_mean, genetic_std = mean_std(genetic_finals)
    random_mean, random_std = mean_std(random_finals)
    print(
        f"genetic mean={genetic_mean:.1f} std={genetic_std:.1f} "
        f"solved={success_rate(genetic_targets):.0%} "
        f"| random mean={random_mean:.1f} std={random_std:.1f} "
        f"solved={success_rate(random_targets):.0%} "
        f"| budget={MAX_EVALUATIONS} evaluations per run"
    )
    print(f"genetic evaluations to reach {TARGET_FITNESS:.0f}: {genetic_targets}")
    print(f"random evaluations to reach {TARGET_FITNESS:.0f}: {random_targets}")
    plot_convergence(
        curves={
            "Algoritmo genético (torneo k=3)": np.stack([result.best_curve for result in genetic_results]),
            "Búsqueda aleatoria": np.stack([result.best_curve for result in random_results]),
        },
        evaluations=genetic_results[0].evaluations_curve,
        title="CartPole-v1 (red 4-16-16-2): algoritmo genético frente a búsqueda aleatoria",
        x_label="Número de evaluaciones",
        y_label="Recompensa media (mejor hasta la fecha)",
        output_path=FIGURE_PATH,
    )
    print(f"figure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
