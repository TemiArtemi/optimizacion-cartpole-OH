import functools
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.evaluation.uci_fitness import UciBatchEvaluator, evaluate_config
from src.genetic.engine import EvolutionResult, run_genetic_algorithm
from src.genetic.hyperparams_chromosome import DEFAULT_CONFIG, decode, random_chromosome
from src.genetic.mixed_operators import mixed_mutation
from src.genetic.operators import uniform_crossover
from src.genetic.random_search import run_random_search
from src.genetic.selection import tournament_selection
from src.networks.torch_mlp import MLPConfig, train_and_evaluate
from src.uci.dataset import LetterDataset, load_letter_dataset
from src.uci.preparation import LETTER_CLASSES
from src.utils.metrics import mean_std
from src.utils.plotting import plot_convergence

POPULATION_SIZE = 20
MAX_EVALUATIONS = 240
EPOCHS = 20
WORKERS = 8
MUTATION_RATE = 0.3
SEED_OFFSET = 10_000
TEST_SEED_OFFSET = 20_000
DEFAULT_SEED_OFFSET = 30_000
RUN_SEEDS = (0, 1, 2, 3, 4)
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "letter"
FIGURE_PATH = Path(__file__).resolve().parents[1] / "results" / "figures" / "phase3_ga_vs_random.png"
LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "logs" / "phase3_check.log"


def init_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Samples one random hyperparameter chromosome for the experiment."""
    return random_chromosome(rng)


def evaluate_on_test(
    config: MLPConfig,
    dataset: LetterDataset,
    epochs: int,
    seed: int,
) -> float:
    """Retrains the configuration on train plus validation and returns the test accuracy."""
    train_features = np.concatenate([dataset.train.features, dataset.validation.features])
    train_labels = np.concatenate([dataset.train.labels, dataset.validation.labels])
    return train_and_evaluate(
        config,
        train_features,
        train_labels,
        dataset.test.features,
        dataset.test.labels,
        class_count=LETTER_CLASSES,
        epochs=epochs,
        seed=seed,
    )


def warm_up_evaluator(evaluator: UciBatchEvaluator, workers: int) -> None:
    """Runs one small untimed batch so every worker process is already running."""
    rng = np.random.default_rng(0)
    population = np.stack([random_chromosome(rng) for _ in range(workers)])
    evaluator.warm_up(population)


def run_genetic(
    run_seed: int,
    data_directory: Path,
    population_size: int,
    max_evaluations: int,
    epochs: int,
    workers: int,
) -> tuple[EvolutionResult, float]:
    """Runs the genetic algorithm for one seed on a parallel evaluator and times it."""
    mutation = functools.partial(mixed_mutation, rate=MUTATION_RATE)
    with UciBatchEvaluator(
        data_directory,
        epochs=epochs,
        workers=workers,
        seed=run_seed + SEED_OFFSET,
    ) as evaluator:
        warm_up_evaluator(evaluator, workers)
        start = time.perf_counter()
        result = run_genetic_algorithm(
            init_fn=init_chromosome,
            selection_fn=tournament_selection,
            crossover_fn=uniform_crossover,
            mutation_fn=mutation,
            population_size=population_size,
            max_evaluations=max_evaluations,
            population_fitness_fn=evaluator,
            seed=run_seed,
        )
        return result, time.perf_counter() - start


def run_random(
    run_seed: int,
    data_directory: Path,
    population_size: int,
    max_evaluations: int,
    epochs: int,
    workers: int,
) -> tuple[EvolutionResult, float]:
    """Runs random search for one seed on a parallel evaluator and times it."""
    with UciBatchEvaluator(
        data_directory,
        epochs=epochs,
        workers=workers,
        seed=run_seed + SEED_OFFSET,
    ) as evaluator:
        warm_up_evaluator(evaluator, workers)
        start = time.perf_counter()
        result = run_random_search(
            None,
            init_chromosome,
            population_size=population_size,
            max_evaluations=max_evaluations,
            seed=run_seed,
            population_fitness_fn=evaluator,
        )
        return result, time.perf_counter() - start


def run_experiment(
    run_seeds: tuple[int, ...],
    population_size: int,
    max_evaluations: int,
    epochs: int,
    workers: int,
    data_directory: Path,
    figure_path: Path,
    log_path: Path,
) -> None:
    """Compares the genetic algorithm, random search and the default configuration."""
    dataset = load_letter_dataset(data_directory)
    log_lines = [
        "dataset=letter_recognition",
        f"population_size={population_size}",
        f"max_evaluations={max_evaluations}",
        f"epochs={epochs}",
        f"mutation_rate={MUTATION_RATE}",
        f"run_seeds={list(run_seeds)}",
        f"workers={workers}",
        "default_config_evaluations=1 per seed",
    ]
    genetic_results: list[EvolutionResult] = []
    random_results: list[EvolutionResult] = []
    default_validation: list[float] = []
    genetic_test: list[float] = []
    random_test: list[float] = []
    default_test: list[float] = []
    genetic_times: list[float] = []
    random_times: list[float] = []
    print(
        f"experiment start: population={population_size} evaluations={max_evaluations} "
        f"epochs={epochs} workers={workers} seeds={list(run_seeds)}"
    )
    for run_seed in run_seeds:
        print(f"seed={run_seed}: genetic search running")
        genetic, genetic_seconds = run_genetic(
            run_seed, data_directory, population_size, max_evaluations, epochs, workers
        )
        print(f"seed={run_seed}: random search running")
        random_result, random_seconds = run_random(
            run_seed, data_directory, population_size, max_evaluations, epochs, workers
        )
        print(f"seed={run_seed}: default and test evaluations running")
        default_validation_accuracy = evaluate_config(
            DEFAULT_CONFIG, dataset, epochs, seed=run_seed + DEFAULT_SEED_OFFSET
        )
        test_seed = run_seed + TEST_SEED_OFFSET
        genetic_test_accuracy = evaluate_on_test(
            decode(genetic.best_chromosome), dataset, epochs, seed=test_seed
        )
        random_test_accuracy = evaluate_on_test(
            decode(random_result.best_chromosome), dataset, epochs, seed=test_seed
        )
        default_test_accuracy = evaluate_on_test(DEFAULT_CONFIG, dataset, epochs, seed=test_seed)
        genetic_results.append(genetic)
        random_results.append(random_result)
        default_validation.append(default_validation_accuracy)
        genetic_test.append(genetic_test_accuracy)
        random_test.append(random_test_accuracy)
        default_test.append(default_test_accuracy)
        genetic_times.append(genetic_seconds)
        random_times.append(random_seconds)
        print(
            f"seed={run_seed}: genetic_val={genetic.best_fitness:.4f} ({genetic_seconds:.1f}s) "
            f"random_val={random_result.best_fitness:.4f} ({random_seconds:.1f}s) "
            f"default_val={default_validation_accuracy:.4f} | genetic_test={genetic_test_accuracy:.4f} "
            f"random_test={random_test_accuracy:.4f} default_test={default_test_accuracy:.4f}"
        )
        log_lines.append(
            f"seed={run_seed}.genetic_val={genetic.best_fitness:.4f}.random_val={random_result.best_fitness:.4f}"
            f".default_val={default_validation_accuracy:.4f}.genetic_test={genetic_test_accuracy:.4f}"
            f".random_test={random_test_accuracy:.4f}.default_test={default_test_accuracy:.4f}"
        )
    genetic_validation_finals = np.array(
        [result.best_fitness for result in genetic_results], dtype=np.float64
    )
    random_validation_finals = np.array(
        [result.best_fitness for result in random_results], dtype=np.float64
    )
    summaries = (
        ("genetic_val", genetic_validation_finals),
        ("random_val", random_validation_finals),
        ("default_val", np.array(default_validation, dtype=np.float64)),
        ("genetic_test", np.array(genetic_test, dtype=np.float64)),
        ("random_test", np.array(random_test, dtype=np.float64)),
        ("default_test", np.array(default_test, dtype=np.float64)),
    )
    for name, values in summaries:
        value_mean, value_std = mean_std(values)
        print(f"{name}: mean={value_mean:.4f} std={value_std:.4f}")
        log_lines.append(f"{name}.mean={value_mean:.4f}.std={value_std:.4f}")
    genetic_time_mean, _ = mean_std(np.array(genetic_times, dtype=np.float64))
    random_time_mean, _ = mean_std(np.array(random_times, dtype=np.float64))
    print(f"mean search time: genetic={genetic_time_mean:.1f}s random={random_time_mean:.1f}s")
    log_lines.append(
        f"search_time.genetic_mean={genetic_time_mean:.1f}.random_mean={random_time_mean:.1f}"
    )
    evaluations = genetic_results[0].evaluations_curve
    plot_convergence(
        curves={
            "Algoritmo genético (torneo k=3)": np.stack(
                [result.best_curve for result in genetic_results]
            ),
            "Búsqueda aleatoria": np.stack([result.best_curve for result in random_results]),
            "Configuración por defecto (media)": np.full(
                (len(run_seeds), len(evaluations)), float(np.mean(default_validation))
            ),
        },
        evaluations=evaluations,
        title="Letter Recognition: AG frente a búsqueda aleatoria y configuración por defecto",
        x_label="Número de evaluaciones",
        y_label="Accuracy en validación (mejor hasta la fecha)",
        output_path=figure_path,
    )
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"figure saved to {figure_path}")
    print(f"log saved to {log_path}")


def main() -> None:
    """Runs the phase 3 comparison with the documented budget and saves the artifacts."""
    run_experiment(
        run_seeds=RUN_SEEDS,
        population_size=POPULATION_SIZE,
        max_evaluations=MAX_EVALUATIONS,
        epochs=EPOCHS,
        workers=WORKERS,
        data_directory=DATA_DIRECTORY,
        figure_path=FIGURE_PATH,
        log_path=LOG_PATH,
    )


if __name__ == "__main__":
    main()
