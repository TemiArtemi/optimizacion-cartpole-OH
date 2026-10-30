import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.baselines.reinforce import run_reinforce
from src.environments.cartpole_eval import CartpoleBatchEvaluator, evaluate_chromosome
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
EPISODE_BUDGET = MAX_EVALUATIONS * EPISODES_PER_EVALUATION
REINFORCE_LEARNING_RATE = 0.001
REINFORCE_BATCH_SIZE = 10
TARGET_FITNESS = 500.0
SEED_OFFSET = 10_000
RUN_SEEDS = (0, 1, 2, 3, 4)
WORKERS = 8
FIGURE_PATH = Path(__file__).resolve().parents[1] / "results" / "figures" / "reinforce_comparison.png"
LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "logs" / "reinforce_comparison.log"


def init_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Samples one random network chromosome for the experiment architecture."""
    return random_chromosome(EXPERIMENT_ARCHITECTURE, rng)


def run_genetic_for_seed(run_seed: int) -> EvolutionResult:
    """Runs the genetic algorithm for one seed on the parallel evaluator."""
    with CartpoleBatchEvaluator(
        architecture=EXPERIMENT_ARCHITECTURE,
        episodes=EPISODES_PER_EVALUATION,
        workers=WORKERS,
        seed=run_seed + SEED_OFFSET,
    ) as evaluator:
        return run_genetic_algorithm(
            init_fn=init_chromosome,
            selection_fn=tournament_selection,
            crossover_fn=one_point_crossover,
            mutation_fn=gaussian_mutation,
            population_size=POPULATION_SIZE,
            max_evaluations=MAX_EVALUATIONS,
            population_fitness_fn=evaluator,
            seed=run_seed,
        )


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


def run_random_for_seed(run_seed: int) -> EvolutionResult:
    """Runs random search for one seed with the same episode budget."""
    return run_random_search(
        fitness_fn=make_fitness(run_seed),
        init_fn=init_chromosome,
        population_size=POPULATION_SIZE,
        max_evaluations=MAX_EVALUATIONS,
        seed=run_seed,
    )


def run_reinforce_for_seed(run_seed: int) -> np.ndarray:
    """Trains REINFORCE for one seed and returns the per-episode return history."""
    result = run_reinforce(
        EXPERIMENT_ARCHITECTURE,
        episodes=EPISODE_BUDGET,
        learning_rate=REINFORCE_LEARNING_RATE,
        batch_size=REINFORCE_BATCH_SIZE,
        seed=run_seed,
    )
    return result.returns


def expand_generation_curve(result: EvolutionResult) -> np.ndarray:
    """Repeats each generation best across the episodes it consumed, one value per episode."""
    return np.repeat(result.best_curve, POPULATION_SIZE * EPISODES_PER_EVALUATION).astype(np.float64)


def episode_axis() -> np.ndarray:
    """Returns the common horizontal axis in episodes played."""
    return np.arange(1, EPISODE_BUDGET + 1, dtype=np.int64)


def summarize(
    method: str,
    finals: list[float],
    targets: list[int | None],
    log_lines: list[str],
) -> None:
    """Prints and logs the final fitness and episode statistics of one method."""
    mean_value, deviation = mean_std(np.array(finals, dtype=np.float64))
    solved = success_rate(targets)
    solved_values = [value for value in targets if value is not None]
    if solved_values:
        target_mean, target_std = mean_std(np.array(solved_values, dtype=np.float64))
        target_text = f"target_mean={target_mean:.1f} target_std={target_std:.1f}"
        target_log = f".target_mean={target_mean:.2f}.target_std={target_std:.2f}"
    else:
        target_text = "target_mean=None target_std=None"
        target_log = ".target_mean=None.target_std=None"
    print(f"{method}: mean={mean_value:.1f} std={deviation:.1f} solved={solved:.0%} {target_text}")
    log_lines.append(
        f"{method}.mean={mean_value:.2f}.std={deviation:.2f}.solved={solved:.2f}{target_log}"
    )


def main() -> None:
    """Compares genetic algorithm, random search and reinforce under one episode budget."""
    axes = episode_axis()
    log_lines = [
        f"architecture={EXPERIMENT_ARCHITECTURE}",
        f"episode_budget={EPISODE_BUDGET}",
        f"run_seeds={list(RUN_SEEDS)}",
        f"reinforce_learning_rate={REINFORCE_LEARNING_RATE}",
        f"reinforce_batch_size={REINFORCE_BATCH_SIZE}",
    ]
    genetic_results: list[EvolutionResult] = []
    for run_seed in RUN_SEEDS:
        start = time.perf_counter()
        result = run_genetic_for_seed(run_seed)
        genetic_results.append(result)
        print(f"genetic seed={run_seed} fitness={result.best_fitness:.1f} ({time.perf_counter() - start:.1f}s)")
        log_lines.append(f"genetic.seed={run_seed}.fitness={result.best_fitness:.2f}")
    random_results: list[EvolutionResult] = []
    for run_seed in RUN_SEEDS:
        start = time.perf_counter()
        result = run_random_for_seed(run_seed)
        random_results.append(result)
        print(f"random seed={run_seed} fitness={result.best_fitness:.1f} ({time.perf_counter() - start:.1f}s)")
        log_lines.append(f"random.seed={run_seed}.fitness={result.best_fitness:.2f}")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(RUN_SEEDS)) as executor:
        reinforce_returns = list(executor.map(run_reinforce_for_seed, RUN_SEEDS))
    print(f"reinforce runs finished ({time.perf_counter() - start:.1f}s)")
    reinforce_curves = [np.maximum.accumulate(returns) for returns in reinforce_returns]
    for run_seed, returns in zip(RUN_SEEDS, reinforce_returns):
        log_lines.append(f"reinforce.seed={run_seed}.best={float(np.max(returns)):.2f}")
    genetic_curves = np.stack([expand_generation_curve(result) for result in genetic_results])
    random_curves = np.stack([expand_generation_curve(result) for result in random_results])
    reinforce_curves_matrix = np.stack(reinforce_curves)
    methods = (
        ("genetic", genetic_curves, [result.best_fitness for result in genetic_results]),
        ("random", random_curves, [result.best_fitness for result in random_results]),
        ("reinforce", reinforce_curves_matrix, [float(np.max(returns)) for returns in reinforce_returns]),
    )
    for name, curves, finals in methods:
        targets = [evaluations_to_target(curve, axes, TARGET_FITNESS) for curve in curves]
        summarize(name, finals, targets, log_lines)
    plot_convergence(
        curves={
            "Algoritmo genético (torneo k=3)": genetic_curves,
            "Búsqueda aleatoria": random_curves,
            "REINFORCE (tasa 0.001)": reinforce_curves_matrix,
        },
        evaluations=axes,
        title="CartPole-v1 (red 4-16-16-2): comparación con igual presupuesto de episodios",
        x_label="Episodios jugados",
        y_label="Recompensa (mejor hasta la fecha)",
        output_path=FIGURE_PATH,
    )
    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"figure saved to {FIGURE_PATH}")
    print(f"log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
