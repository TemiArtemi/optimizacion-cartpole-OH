import functools

import numpy as np

from src.environments.cartpole_eval import (
    CARTPOLE_ARCHITECTURE,
    CartpoleBatchEvaluator,
    EvaluationTask,
    draw_evaluation_seeds,
    evaluate_chromosome,
    evaluate_task,
)
from src.genetic.engine import run_genetic_algorithm
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.genetic.selection import tournament_selection
from src.networks.numpy_mlp import random_chromosome


def sample_population(count: int, seed: int) -> np.ndarray:
    """Builds a fixed population of random chromosomes for the test."""
    rng = np.random.default_rng(seed)
    return np.stack([random_chromosome(CARTPOLE_ARCHITECTURE, rng) for _ in range(count)])


def init_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Samples one random chromosome for the engine test."""
    return random_chromosome(CARTPOLE_ARCHITECTURE, rng)


def test_warm_up_does_not_consume_the_run_seed() -> None:
    """Checks that warming up the pool leaves the timed seed sequence untouched."""
    population = sample_population(4, 6)
    seeds = draw_evaluation_seeds(np.random.default_rng(10), 4)
    serial = np.array(
        [
            evaluate_chromosome(chromosome, episodes=1, seed=int(seeds[index]))
            for index, chromosome in enumerate(population)
        ]
    )
    with CartpoleBatchEvaluator(episodes=1, workers=2, seed=10) as evaluator:
        evaluator.warm_up(population)
        scores = evaluator(population)
    assert np.array_equal(serial, scores)


def test_worker_matches_serial_evaluation() -> None:
    """Checks that evaluating a task on a worker equals calling the serial evaluator."""
    chromosome = random_chromosome(CARTPOLE_ARCHITECTURE, np.random.default_rng(0))
    task = EvaluationTask(chromosome=chromosome, architecture=CARTPOLE_ARCHITECTURE, episodes=1, seed=11)
    assert evaluate_task(task) == evaluate_chromosome(chromosome, episodes=1, seed=11)


def test_batch_matches_serial_with_preassigned_seeds() -> None:
    """Checks that the parallel batch equals a serial loop using the same assigned seeds."""
    population = sample_population(4, 1)
    seeds = draw_evaluation_seeds(np.random.default_rng(7), 4)
    serial = np.array(
        [
            evaluate_chromosome(chromosome, episodes=1, seed=int(seeds[index]))
            for index, chromosome in enumerate(population)
        ]
    )
    with CartpoleBatchEvaluator(episodes=1, workers=2, seed=7) as evaluator:
        parallel = evaluator(population)
    assert parallel.shape == (4,)
    assert np.array_equal(serial, parallel)


def test_batch_is_independent_of_worker_count() -> None:
    """Checks that the same run seed produces identical results for any worker count."""
    population = sample_population(4, 2)
    with CartpoleBatchEvaluator(episodes=1, workers=1, seed=8) as single_worker:
        first = single_worker(population)
    with CartpoleBatchEvaluator(episodes=1, workers=3, seed=8) as many_workers:
        second = many_workers(population)
    assert np.array_equal(first, second)


def test_batch_is_reproducible_for_a_fixed_seed() -> None:
    """Checks that two evaluators built with the same seed score the population identically."""
    population = sample_population(4, 3)
    with CartpoleBatchEvaluator(episodes=1, workers=2, seed=9) as first_evaluator:
        first = first_evaluator(population)
    with CartpoleBatchEvaluator(episodes=1, workers=2, seed=9) as second_evaluator:
        second = second_evaluator(population)
    assert np.array_equal(first, second)


def test_evaluator_rejects_invalid_configuration() -> None:
    """Checks that non-positive worker and episode counts are rejected."""
    try:
        CartpoleBatchEvaluator(workers=0)
        raise AssertionError("expected ValueError for zero workers")
    except ValueError:
        pass
    try:
        CartpoleBatchEvaluator(episodes=0)
        raise AssertionError("expected ValueError for zero episodes")
    except ValueError:
        pass


def engine_arguments(**extra: object) -> dict[str, object]:
    """Collects the shared engine arguments used by the integration tests."""
    return {
        "init_fn": init_chromosome,
        "selection_fn": tournament_selection,
        "crossover_fn": one_point_crossover,
        "mutation_fn": functools.partial(gaussian_mutation, rate=0.1, sigma=0.1),
        "population_size": 10,
        "max_evaluations": 30,
        "seed": 2,
        **extra,
    }


def test_engine_runs_with_population_scorer() -> None:
    """Checks that the engine works with the parallel batch scorer and stays reproducible."""
    with CartpoleBatchEvaluator(episodes=1, workers=2, seed=4) as first_evaluator:
        first = run_genetic_algorithm(population_fitness_fn=first_evaluator, **engine_arguments())
    with CartpoleBatchEvaluator(episodes=1, workers=2, seed=4) as second_evaluator:
        second = run_genetic_algorithm(population_fitness_fn=second_evaluator, **engine_arguments())
    assert first.history[-1].evaluations == 30
    assert first.best_fitness == second.best_fitness
    assert np.array_equal(first.best_chromosome, second.best_chromosome)


def test_engine_rejects_scorer_configuration() -> None:
    """Checks that providing zero or two scoring functions is rejected."""
    try:
        run_genetic_algorithm(**engine_arguments())
        raise AssertionError("expected ValueError when no scoring function is provided")
    except ValueError:
        pass
    with CartpoleBatchEvaluator(episodes=1, workers=1, seed=5) as evaluator:
        try:
            run_genetic_algorithm(
                fitness_fn=lambda chromosome: 0.0,
                population_fitness_fn=evaluator,
                **engine_arguments(),
            )
            raise AssertionError("expected ValueError when both scoring functions are provided")
        except ValueError:
            pass


def main() -> None:
    """Runs every parallel evaluation test and reports the result."""
    test_warm_up_does_not_consume_the_run_seed()
    test_worker_matches_serial_evaluation()
    test_batch_matches_serial_with_preassigned_seeds()
    test_batch_is_independent_of_worker_count()
    test_batch_is_reproducible_for_a_fixed_seed()
    test_evaluator_rejects_invalid_configuration()
    test_engine_runs_with_population_scorer()
    test_engine_rejects_scorer_configuration()
    print("All parallel evaluation tests passed.")


if __name__ == "__main__":
    main()
