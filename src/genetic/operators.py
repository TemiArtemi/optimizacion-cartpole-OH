import numpy as np
from numpy.typing import NDArray


def one_point_crossover(
    parent_a: NDArray[np.float64],
    parent_b: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Splits two parent chromosomes at one random point and swaps their tails."""
    point = int(rng.integers(1, len(parent_a)))
    child_a = np.concatenate((parent_a[:point], parent_b[point:]))
    child_b = np.concatenate((parent_b[:point], parent_a[point:]))
    return child_a, child_b


def gaussian_mutation(
    chromosome: NDArray[np.float64],
    rng: np.random.Generator,
    rate: float = 0.1,
    sigma: float = 0.1,
) -> NDArray[np.float64]:
    """Returns a copy of the chromosome with per-gene Gaussian noise."""
    mutated = chromosome.copy()
    mask = rng.random(len(mutated)) < rate
    mutated[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))
    return mutated
