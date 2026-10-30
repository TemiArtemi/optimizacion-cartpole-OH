import numpy as np
from numpy.typing import NDArray


def _validate_non_empty(fitnesses: NDArray[np.float64]) -> int:
    """Returns the population size or raises when the fitness array is empty."""
    population_size = int(fitnesses.shape[0])
    if population_size == 0:
        raise ValueError("Cannot select from an empty fitness array.")
    return population_size


def tournament_selection(
    fitnesses: NDArray[np.float64],
    rng: np.random.Generator,
    selected_count: int,
    tournament_size: int = 3,
) -> NDArray[np.int64]:
    """Selects parent indices by keeping the fittest member of each random sample."""
    population_size = _validate_non_empty(fitnesses)
    if not 1 <= tournament_size <= population_size:
        raise ValueError("Tournament size must be between 1 and the population size.")
    selected = np.empty(selected_count, dtype=np.int64)
    for slot in range(selected_count):
        contestants = rng.choice(population_size, size=tournament_size, replace=False)
        selected[slot] = contestants[int(np.argmax(fitnesses[contestants]))]
    return selected


def roulette_wheel_selection(
    fitnesses: NDArray[np.float64],
    rng: np.random.Generator,
    selected_count: int,
) -> NDArray[np.int64]:
    """Selects parent indices with probability proportional to shifted fitness."""
    population_size = _validate_non_empty(fitnesses)
    shifted = fitnesses - float(np.min(fitnesses)) + 1e-12
    probabilities = shifted / float(np.sum(shifted))
    selected = rng.choice(population_size, size=selected_count, replace=True, p=probabilities)
    return selected.astype(np.int64)
