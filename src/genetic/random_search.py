import numpy as np
from numpy.typing import NDArray

from src.genetic.engine import (
    EvolutionResult,
    FitnessFunction,
    GenerationStats,
    InitializerFunction,
)


def run_random_search(
    fitness_fn: FitnessFunction,
    init_fn: InitializerFunction,
    population_size: int,
    max_evaluations: int,
    seed: int | None = None,
) -> EvolutionResult:
    """Evaluates random batches under a fixed budget and returns the best solution."""
    if population_size < 1:
        raise ValueError("Population size must be positive.")
    if max_evaluations < population_size:
        raise ValueError("The evaluation budget must fit at least one full batch.")
    rng = np.random.default_rng(seed)
    generations = max_evaluations // population_size
    history: list[GenerationStats] = []
    best_chromosome: NDArray[np.float64] | None = None
    best_fitness = -float(np.inf)
    for generation in range(generations):
        batch = np.stack([init_fn(rng) for _ in range(population_size)])
        fitnesses = np.array([fitness_fn(individual) for individual in batch], dtype=np.float64)
        order = np.argsort(fitnesses)[::-1]
        if float(fitnesses[order[0]]) > best_fitness:
            best_fitness = float(fitnesses[order[0]])
            best_chromosome = batch[order[0]].copy()
        history.append(
            GenerationStats(
                generation=generation,
                best_so_far=best_fitness,
                mean=float(np.mean(fitnesses)),
                std=float(np.std(fitnesses)),
                evaluations=(generation + 1) * population_size,
            )
        )
    return EvolutionResult(
        best_chromosome=best_chromosome,
        best_fitness=best_fitness,
        history=history,
    )
