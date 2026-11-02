import numpy as np
from numpy.typing import NDArray

from src.genetic.hyperparams_chromosome import (
    CHROMOSOME_LENGTH,
    MUTATION_SIGMA,
    clip_chromosome,
    mutate_gene,
)


def mixed_mutation(
    chromosome: NDArray[np.float64],
    rng: np.random.Generator,
    rate: float = 0.2,
    sigma: float = MUTATION_SIGMA,
) -> NDArray[np.float64]:
    """Returns a clipped copy where each gene mutates with type-aware noise at the given rate."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("Mutation rate must lie between 0 and 1.")
    if sigma < 0.0:
        raise ValueError("Sigma must not be negative.")
    mutated = clip_chromosome(chromosome)
    for gene in range(CHROMOSOME_LENGTH):
        if rng.random() < rate:
            mutated[gene] = mutate_gene(gene, float(mutated[gene]), rng, sigma)
    return mutated
