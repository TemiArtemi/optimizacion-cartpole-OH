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


def uniform_crossover(
    parent_a: NDArray[np.float64],
    parent_b: NDArray[np.float64],
    rng: np.random.Generator,
    swap_probability: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Builds two children by taking each gene from either parent with the given probability."""
    if not 0.0 <= swap_probability <= 1.0:
        raise ValueError("Swap probability must lie between 0 and 1.")
    swap = rng.random(len(parent_a)) < swap_probability
    child_a = np.where(swap, parent_b, parent_a)
    child_b = np.where(swap, parent_a, parent_b)
    return child_a.copy(), child_b.copy()


def blx_alpha_crossover(
    parent_a: NDArray[np.float64],
    parent_b: NDArray[np.float64],
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Samples each child gene uniformly from the parents' range widened by alpha times the gap."""
    if alpha < 0.0:
        raise ValueError("Alpha must not be negative.")
    low = np.minimum(parent_a, parent_b)
    high = np.maximum(parent_a, parent_b)
    span = (high - low) * alpha
    lower_bounds = low - span
    upper_bounds = high + span
    child_a = rng.uniform(lower_bounds, upper_bounds)
    child_b = rng.uniform(lower_bounds, upper_bounds)
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
