from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FitnessFunction = Callable[[NDArray[np.float64]], float]
InitializerFunction = Callable[[np.random.Generator], NDArray[np.float64]]
SelectionFunction = Callable[
    [NDArray[np.float64], np.random.Generator, int], NDArray[np.int64]
]
CrossoverFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64], np.random.Generator],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]
MutationFunction = Callable[[NDArray[np.float64], np.random.Generator], NDArray[np.float64]]


@dataclass(frozen=True)
class GenerationStats:
    """Fitness summary of one evaluated generation."""
    generation: int
    best_so_far: float
    mean: float
    std: float
    evaluations: int


@dataclass(frozen=True)
class EvolutionResult:
    """Best solution found plus the convergence history of the run."""
    best_chromosome: NDArray[np.float64]
    best_fitness: float
    history: list[GenerationStats]


def run_genetic_algorithm(
    fitness_fn: FitnessFunction,
    init_fn: InitializerFunction,
    selection_fn: SelectionFunction,
    crossover_fn: CrossoverFunction,
    mutation_fn: MutationFunction,
    population_size: int,
    max_evaluations: int,
    crossover_rate: float = 0.9,
    elite_size: int = 2,
    seed: int | None = None,
) -> EvolutionResult:
    """Evolves a population under a fixed evaluation budget and returns the best solution."""
    if population_size < 2:
        raise ValueError("Population size must be at least 2.")
    if max_evaluations < population_size:
        raise ValueError("The evaluation budget must fit at least one full generation.")
    if not 0 <= elite_size < population_size:
        raise ValueError("Elite size must be between 0 and population size minus one.")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("Crossover rate must lie between 0 and 1.")
    rng = np.random.default_rng(seed)
    generations = max_evaluations // population_size
    population = np.stack([init_fn(rng) for _ in range(population_size)])
    history: list[GenerationStats] = []
    best_chromosome = population[0].copy()
    best_fitness = -float(np.inf)
    for generation in range(generations):
        fitnesses = np.array([fitness_fn(individual) for individual in population], dtype=np.float64)
        order = np.argsort(fitnesses)[::-1]
        if float(fitnesses[order[0]]) > best_fitness:
            best_fitness = float(fitnesses[order[0]])
            best_chromosome = population[order[0]].copy()
        history.append(
            GenerationStats(
                generation=generation,
                best_so_far=best_fitness,
                mean=float(np.mean(fitnesses)),
                std=float(np.std(fitnesses)),
                evaluations=(generation + 1) * population_size,
            )
        )
        elites = population[order[:elite_size]]
        parent_indices = selection_fn(fitnesses, rng, population_size - elite_size)
        children: list[NDArray[np.float64]] = []
        cursor = 0
        while len(children) < population_size - elite_size:
            parent_a = population[parent_indices[cursor % len(parent_indices)]]
            parent_b = population[parent_indices[(cursor + 1) % len(parent_indices)]]
            cursor += 2
            if rng.random() < crossover_rate:
                child_a, child_b = crossover_fn(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
            children.append(mutation_fn(child_a, rng))
            children.append(mutation_fn(child_b, rng))
        population = np.vstack((elites, np.stack(children[: population_size - elite_size])))
    return EvolutionResult(
        best_chromosome=best_chromosome,
        best_fitness=best_fitness,
        history=history,
    )
