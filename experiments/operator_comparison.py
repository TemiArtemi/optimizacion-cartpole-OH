import functools
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.environments.cartpole_eval import CartpoleBatchEvaluator
from src.genetic.engine import (
    CrossoverFunction,
    EvolutionResult,
    SelectionFunction,
    run_genetic_algorithm,
)
from src.genetic.operators import (
    blx_alpha_crossover,
    gaussian_mutation,
    one_point_crossover,
    uniform_crossover,
)
from src.genetic.selection import roulette_wheel_selection, tournament_selection
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
WORKERS = 8
FIGURE_PATH = Path(__file__).resolve().parents[1] / "results" / "figures" / "operator_comparison.png"
LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "logs" / "operator_comparison.log"


@dataclass(frozen=True)
class Configuration:
    """One combination of selection and crossover operators under comparison."""
    identifier: str
    display_label: str
    selection_fn: SelectionFunction
    crossover_fn: CrossoverFunction


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration(
        "tournament_one_point",
        "Torneo + cruce un punto",
        tournament_selection,
        one_point_crossover,
    ),
    Configuration(
        "roulette_one_point",
        "Ruleta + cruce un punto",
        roulette_wheel_selection,
        one_point_crossover,
    ),
    Configuration(
        "tournament_uniform",
        "Torneo + cruce uniforme",
        tournament_selection,
        uniform_crossover,
    ),
    Configuration(
        "tournament_blx_alpha",
        "Torneo + cruce BLX-α",
        tournament_selection,
        blx_alpha_crossover,
    ),
)


def init_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Samples one random network chromosome for the experiment architecture."""
    return random_chromosome(EXPERIMENT_ARCHITECTURE, rng)


def run_configuration(config: Configuration, run_seed: int) -> tuple[EvolutionResult, float]:
    """Runs one configuration with one seed on the parallel evaluator and times it."""
    mutation = functools.partial(gaussian_mutation, rate=0.1, sigma=0.1)
    with CartpoleBatchEvaluator(
        architecture=EXPERIMENT_ARCHITECTURE,
        episodes=EPISODES_PER_EVALUATION,
        workers=WORKERS,
        seed=run_seed + SEED_OFFSET,
    ) as evaluator:
        start = time.perf_counter()
        result = run_genetic_algorithm(
            init_fn=init_chromosome,
            selection_fn=config.selection_fn,
            crossover_fn=config.crossover_fn,
            mutation_fn=mutation,
            population_size=POPULATION_SIZE,
            max_evaluations=MAX_EVALUATIONS,
            population_fitness_fn=evaluator,
            seed=run_seed,
        )
        return result, time.perf_counter() - start


def target_statistics(targets: list[int | None]) -> tuple[float | None, float | None]:
    """Returns mean and sample deviation of the evaluations to target over solved runs only."""
    solved = [value for value in targets if value is not None]
    if not solved:
        return None, None
    mean_value, deviation = mean_std(np.array(solved, dtype=np.float64))
    return mean_value, deviation


def main() -> None:
    """Compares the operator configurations on CartPole and saves figure and log."""
    log_lines = [
        f"architecture={EXPERIMENT_ARCHITECTURE}",
        f"population_size={POPULATION_SIZE}",
        f"max_evaluations={MAX_EVALUATIONS}",
        f"run_seeds={list(RUN_SEEDS)}",
        f"workers={WORKERS}",
    ]
    curves: dict[str, np.ndarray] = {}
    last_results: list[EvolutionResult] = []
    for config in CONFIGURATIONS:
        results: list[EvolutionResult] = []
        targets: list[int | None] = []
        print(f"configuration={config.identifier}")
        for run_seed in RUN_SEEDS:
            result, seconds = run_configuration(config, run_seed)
            target = evaluations_to_target(result.best_curve, result.evaluations_curve, TARGET_FITNESS)
            results.append(result)
            targets.append(target)
            print(
                f"  seed={run_seed} fitness={result.best_fitness:.1f} "
                f"at={target} ({seconds:.1f}s)"
            )
            log_lines.append(
                f"{config.identifier}.seed={run_seed}.fitness={result.best_fitness:.2f}.target={target}"
            )
        finals = np.array([result.best_fitness for result in results], dtype=np.float64)
        fitness_mean, fitness_std = mean_std(finals)
        target_mean, target_std = target_statistics(targets)
        solved = success_rate(targets)
        print(
            f"  mean={fitness_mean:.1f} std={fitness_std:.1f} solved={solved:.0%} "
            f"target_mean={target_mean if target_mean is None else round(target_mean, 1)} "
            f"target_std={target_std if target_std is None else round(target_std, 1)}"
        )
        log_lines.append(
            f"{config.identifier}.mean={fitness_mean:.2f}.std={fitness_std:.2f}"
            f".solved={solved:.2f}.target_mean={target_mean}.target_std={target_std}"
        )
        curves[config.display_label] = np.stack([result.best_curve for result in results])
        last_results = results
    plot_convergence(
        curves=curves,
        evaluations=last_results[0].evaluations_curve,
        title="CartPole-v1 (red 4-16-16-2): comparación de operadores genéticos",
        x_label="Número de evaluaciones",
        y_label="Recompensa media (mejor hasta la fecha)",
        output_path=FIGURE_PATH,
    )
    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"figure saved to {FIGURE_PATH}")
    print(f"log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
