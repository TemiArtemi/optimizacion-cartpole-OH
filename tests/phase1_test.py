import functools

import numpy as np

from src.genetic.engine import run_genetic_algorithm
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.genetic.random_search import run_random_search
from src.genetic.selection import roulette_wheel_selection, tournament_selection

SPHERE_DIMENSION = 10


def sphere_init(rng: np.random.Generator) -> np.ndarray:
    """Samples one individual for the sphere benchmark within the search box."""
    return rng.uniform(-2.0, 2.0, size=SPHERE_DIMENSION)


def sphere_fitness(chromosome: np.ndarray) -> float:
    """Returns the sphere benchmark value to maximize."""
    return -float(np.sum(np.asarray(chromosome, dtype=np.float64) ** 2))


def sphere_mutation() -> object:
    """Builds the benchmark mutation operator with fixed rate and sigma."""
    return functools.partial(gaussian_mutation, rate=0.1, sigma=0.1)


def test_tournament_returns_fittest_when_sampling_whole_field() -> None:
    """Checks that a tournament covering the population always picks the best individual."""
    fitnesses = np.array([0.0, 5.0, 0.0, 1.0])
    selected = tournament_selection(fitnesses, np.random.default_rng(0), 25, tournament_size=4)
    assert selected.shape == (25,)
    assert np.all(selected == 1)


def test_tournament_rejects_invalid_tournament_size() -> None:
    """Checks that tournament sizes outside the valid range are rejected."""
    fitnesses = np.array([1.0, 2.0, 3.0])
    try:
        tournament_selection(fitnesses, np.random.default_rng(0), 3, tournament_size=4)
        raise AssertionError("expected ValueError for tournament larger than population")
    except ValueError:
        pass
    try:
        tournament_selection(fitnesses, np.random.default_rng(0), 3, tournament_size=0)
        raise AssertionError("expected ValueError for empty tournament")
    except ValueError:
        pass


def test_roulette_dominates_with_shifted_negative_fitness() -> None:
    """Checks that roulette handles negative fitness and favors the fittest individual."""
    fitnesses = np.array([-10.0, -1.0, -5.0])
    selected = roulette_wheel_selection(fitnesses, np.random.default_rng(1), 4000)
    assert selected.shape == (4000,)
    assert np.all((0 <= selected) & (selected < 3))
    counts = np.bincount(selected, minlength=3) / 4000.0
    assert counts[1] > 0.5
    assert counts[0] < 0.01


def test_selection_operators_do_not_mutate_inputs() -> None:
    """Checks that both selectors leave the fitness array untouched."""
    fitnesses = np.array([1.0, 3.0, 2.0, 0.5])
    original = fitnesses.copy()
    tournament_selection(fitnesses, np.random.default_rng(2), 10)
    assert np.array_equal(fitnesses, original)
    roulette_wheel_selection(fitnesses, np.random.default_rng(2), 10)
    assert np.array_equal(fitnesses, original)


def test_engine_respects_budget_and_keeps_best_monotone() -> None:
    """Checks that the engine consumes the exact budget and never loses its best solution."""
    result = run_genetic_algorithm(
        fitness_fn=sphere_fitness,
        init_fn=sphere_init,
        selection_fn=tournament_selection,
        crossover_fn=one_point_crossover,
        mutation_fn=sphere_mutation(),
        population_size=30,
        max_evaluations=150,
        seed=3,
    )
    assert result.history[-1].evaluations == 150
    assert len(result.history) == 5
    assert np.all(np.diff([entry.best_so_far for entry in result.history]) >= 0.0)
    assert result.best_fitness == result.history[-1].best_so_far


def test_engine_is_reproducible_for_a_fixed_seed() -> None:
    """Checks that two identical runs produce identical results and histories."""
    parameters = {
        "fitness_fn": sphere_fitness,
        "init_fn": sphere_init,
        "selection_fn": tournament_selection,
        "crossover_fn": one_point_crossover,
        "mutation_fn": sphere_mutation(),
        "population_size": 30,
        "max_evaluations": 600,
    }
    first = run_genetic_algorithm(seed=5, **parameters)
    second = run_genetic_algorithm(seed=5, **parameters)
    assert first.best_fitness == second.best_fitness
    assert np.array_equal(first.best_chromosome, second.best_chromosome)
    assert [entry.best_so_far for entry in first.history] == [
        entry.best_so_far for entry in second.history
    ]


def test_engine_rejects_invalid_configuration() -> None:
    """Checks that budgets and elite sizes outside the valid ranges are rejected."""
    arguments = {
        "fitness_fn": sphere_fitness,
        "init_fn": sphere_init,
        "selection_fn": tournament_selection,
        "crossover_fn": one_point_crossover,
        "mutation_fn": sphere_mutation(),
        "population_size": 10,
        "max_evaluations": 100,
    }
    try:
        run_genetic_algorithm(population_size=1, max_evaluations=10, **{k: v for k, v in arguments.items() if k not in ("population_size", "max_evaluations")})
        raise AssertionError("expected ValueError for population size below 2")
    except ValueError:
        pass
    try:
        run_genetic_algorithm(population_size=10, max_evaluations=5, **{k: v for k, v in arguments.items() if k not in ("population_size", "max_evaluations")})
        raise AssertionError("expected ValueError for budget below one generation")
    except ValueError:
        pass
    try:
        run_genetic_algorithm(elite_size=10, **arguments)
        raise AssertionError("expected ValueError for elite size equal to population size")
    except ValueError:
        pass


def test_random_search_respects_budget() -> None:
    """Checks that random search consumes the exact budget in whole batches."""
    result = run_random_search(sphere_fitness, sphere_init, population_size=25, max_evaluations=100, seed=6)
    assert result.history[-1].evaluations == 100
    assert len(result.history) == 4
    assert result.best_fitness == result.history[-1].best_so_far


def test_genetic_algorithm_beats_random_search_under_same_budget() -> None:
    """Checks that the genetic algorithm clearly outperforms random search on the sphere."""
    ga = run_genetic_algorithm(
        fitness_fn=sphere_fitness,
        init_fn=sphere_init,
        selection_fn=tournament_selection,
        crossover_fn=one_point_crossover,
        mutation_fn=sphere_mutation(),
        population_size=40,
        max_evaluations=4000,
        seed=7,
    )
    random_baseline = run_random_search(sphere_fitness, sphere_init, population_size=40, max_evaluations=4000, seed=7)
    assert ga.best_fitness > random_baseline.best_fitness
    assert ga.best_fitness > 10.0 * random_baseline.best_fitness


def main() -> None:
    """Runs every phase 1 test and reports the result."""
    test_tournament_returns_fittest_when_sampling_whole_field()
    test_tournament_rejects_invalid_tournament_size()
    test_roulette_dominates_with_shifted_negative_fitness()
    test_selection_operators_do_not_mutate_inputs()
    test_engine_respects_budget_and_keeps_best_monotone()
    test_engine_is_reproducible_for_a_fixed_seed()
    test_engine_rejects_invalid_configuration()
    test_random_search_respects_budget()
    test_genetic_algorithm_beats_random_search_under_same_budget()
    print("All phase 1 tests passed.")


if __name__ == "__main__":
    main()
