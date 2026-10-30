import functools
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from experiments.phase1_check import EXPERIMENT_ARCHITECTURE, init_chromosome, make_fitness
from src.environments.cartpole_eval import (
    CartpoleBatchEvaluator,
    draw_evaluation_seeds,
    evaluate_chromosome,
)
from src.genetic.engine import run_genetic_algorithm
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.genetic.selection import tournament_selection
from src.networks.numpy_mlp import random_chromosome

POPULATION_SIZE = 50
EPISODES_PER_EVALUATION = 3
EVALUATOR_SEED = 123
REPEATS = 5
HEAVY_WARMUP_BUDGET = 600
HEAVY_WARMUP_SEED = 3
FIGURE_PATH = Path(__file__).resolve().parents[1] / "results" / "figures" / "parallel_speedup.png"
LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "logs" / "parallel_speedup.log"


def build_light_population() -> np.ndarray:
    """Builds a random population representing the early generations of a run."""
    rng = np.random.default_rng(2024)
    return np.stack([random_chromosome(EXPERIMENT_ARCHITECTURE, rng) for _ in range(POPULATION_SIZE)])


def build_heavy_population() -> tuple[np.ndarray, float]:
    """Runs a short genetic algorithm and replicates its best chromosome into a converged population."""
    result = run_genetic_algorithm(
        fitness_fn=make_fitness(HEAVY_WARMUP_SEED),
        init_fn=init_chromosome,
        selection_fn=tournament_selection,
        crossover_fn=one_point_crossover,
        mutation_fn=functools.partial(gaussian_mutation, rate=0.1, sigma=0.1),
        population_size=POPULATION_SIZE,
        max_evaluations=HEAVY_WARMUP_BUDGET,
        seed=HEAVY_WARMUP_SEED,
    )
    population = np.tile(result.best_chromosome, (POPULATION_SIZE, 1))
    return population, result.best_fitness


def serial_reference(population: np.ndarray) -> tuple[np.ndarray, float]:
    """Scores the population sequentially and returns the fitness values and the elapsed seconds."""
    seeds = draw_evaluation_seeds(np.random.default_rng(EVALUATOR_SEED), len(population))
    start = time.perf_counter()
    scores = np.array(
        [
            evaluate_chromosome(
                chromosome,
                architecture=EXPERIMENT_ARCHITECTURE,
                episodes=EPISODES_PER_EVALUATION,
                seed=int(seeds[index]),
            )
            for index, chromosome in enumerate(population)
        ]
    )
    return scores, time.perf_counter() - start


def parallel_run(population: np.ndarray, workers: int) -> tuple[np.ndarray, float]:
    """Warms up a fresh pool, then scores the population and returns values with elapsed seconds."""
    with CartpoleBatchEvaluator(
        architecture=EXPERIMENT_ARCHITECTURE,
        episodes=EPISODES_PER_EVALUATION,
        workers=workers,
        seed=EVALUATOR_SEED,
    ) as evaluator:
        evaluator.warm_up(population)
        start = time.perf_counter()
        scores = evaluator(population)
        elapsed = time.perf_counter() - start
    return scores, elapsed


def worker_counts() -> list[int]:
    """Returns the pool sizes to benchmark, capped by the machine core count."""
    core_count = max(os.cpu_count() or 1, 1)
    return [count for count in (1, 2, 4, 8, 12, 16) if count <= core_count]


def benchmark_workload(
    name: str,
    population: np.ndarray,
    counts: list[int],
    log_lines: list[str],
) -> tuple[float, list[float]]:
    """Times one workload sequentially and in parallel, appending raw measurements to the log."""
    reference, _ = serial_reference(population)
    log_lines.append(f"{name}.reference_mean_fitness={reference.mean():.2f}")
    print(f"{name}: reference mean fitness={reference.mean():.1f}")
    serial_times: list[float] = []
    for _ in range(REPEATS):
        scores, seconds = serial_reference(population)
        assert np.array_equal(scores, reference)
        serial_times.append(seconds)
    serial_seconds = float(np.mean(serial_times))
    log_lines.append(f"{name}.serial.times={','.join(f'{value:.4f}' for value in serial_times)}")
    print(f"{name}: serial mean={serial_seconds:.3f}s")
    parallel_seconds: list[float] = []
    for workers in counts:
        times: list[float] = []
        for _ in range(REPEATS):
            scores, seconds = parallel_run(population, workers)
            assert np.array_equal(scores, reference)
            times.append(seconds)
        mean_seconds = float(np.mean(times))
        parallel_seconds.append(mean_seconds)
        log_lines.append(f"{name}.workers={workers}.times={','.join(f'{value:.4f}' for value in times)}")
        print(f"{name}: workers={workers} mean={mean_seconds:.3f}s speedup={serial_seconds / mean_seconds:.2f}x")
    return serial_seconds, parallel_seconds


def plot_results(
    light: tuple[float, list[float]],
    heavy: tuple[float, list[float]],
    counts: list[int],
) -> None:
    """Saves the timing and speedup figure with Spanish labels for both workloads."""
    labels = ["Secuencial"] + [str(count) for count in counts]
    positions = np.arange(len(labels))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(positions - width / 2, [light[0]] + light[1], width, label="Población aleatoria", color="tab:blue")
    axes[0].bar(positions + width / 2, [heavy[0]] + heavy[1], width, label="Población evolucionada", color="tab:green")
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, labels)
    axes[0].set_xlabel("Configuración (procesos)")
    axes[0].set_ylabel("Tiempo medio por población (s, escala logarítmica)")
    axes[0].set_title("Tiempo de evaluación")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].plot(counts, [light[0] / value for value in light[1]], marker="o", label="Población aleatoria", color="tab:blue")
    axes[1].plot(counts, [heavy[0] / value for value in heavy[1]], marker="s", label="Población evolucionada", color="tab:green")
    axes[1].plot(counts, counts, linestyle="--", label="Ideal", color="tab:orange")
    axes[1].set_xlabel("Número de procesos")
    axes[1].set_ylabel("Aceleración (speedup)")
    axes[1].set_title("Aceleración frente al ideal")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    """Measures sequential and parallel evaluation for light and heavy workloads."""
    counts = worker_counts()
    log_lines = [
        f"core_count={os.cpu_count()}",
        f"population_size={POPULATION_SIZE}",
        f"repeats={REPEATS}",
        f"worker_counts={counts}",
    ]
    light_population = build_light_population()
    heavy_population, heavy_fitness = build_heavy_population()
    log_lines.append(f"heavy_warmup_fitness={heavy_fitness:.1f}")
    print(f"heavy warm-up population fitness: {heavy_fitness:.1f}")
    light_results = benchmark_workload("light", light_population, counts, log_lines)
    heavy_results = benchmark_workload("heavy", heavy_population, counts, log_lines)
    for name, results in (("light", light_results), ("heavy", heavy_results)):
        serial_seconds, parallel_seconds = results
        log_lines.append(f"{name}.serial_mean={serial_seconds:.4f}")
        for workers, seconds in zip(counts, parallel_seconds):
            log_lines.append(f"{name}.workers={workers}_mean={seconds:.4f}_speedup={serial_seconds / seconds:.3f}")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    plot_results(light_results, heavy_results, counts)
    print(f"figure saved to {FIGURE_PATH}")
    print(f"log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
