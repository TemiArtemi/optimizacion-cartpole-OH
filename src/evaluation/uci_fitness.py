from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from src.genetic.hyperparams_chromosome import decode
from src.networks.torch_mlp import MLPConfig, train_and_evaluate
from src.uci.dataset import LetterDataset, load_letter_dataset
from src.uci.preparation import LETTER_CLASSES
from src.utils.seeding import draw_evaluation_seeds

_WORKER_DATASET: LetterDataset | None = None


@dataclass(frozen=True)
class FitnessTask:
    """Bundles one chromosome with its evaluation settings for a worker process."""
    chromosome: NDArray[np.float64]
    epochs: int
    seed: int


def evaluate_config(
    config: MLPConfig,
    dataset: LetterDataset,
    epochs: int,
    seed: int,
) -> float:
    """Trains the given configuration and returns its validation accuracy."""
    return train_and_evaluate(
        config,
        dataset.train.features,
        dataset.train.labels,
        dataset.validation.features,
        dataset.validation.labels,
        class_count=LETTER_CLASSES,
        epochs=epochs,
        seed=seed,
    )


def evaluate_chromosome(
    chromosome: NDArray[np.float64],
    dataset: LetterDataset,
    epochs: int,
    seed: int,
) -> float:
    """Trains the encoded configuration and returns its validation accuracy."""
    return evaluate_config(decode(chromosome), dataset, epochs, seed)


def initialize_worker(data_directory: Path) -> None:
    """Loads the letter dataset once inside each worker process."""
    global _WORKER_DATASET
    _WORKER_DATASET = load_letter_dataset(data_directory)


def evaluate_task(task: FitnessTask) -> float:
    """Evaluates one task inside a worker process and returns the validation accuracy."""
    if _WORKER_DATASET is None:
        raise RuntimeError("The worker dataset has not been initialized.")
    return evaluate_chromosome(task.chromosome, _WORKER_DATASET, task.epochs, task.seed)


class UciBatchEvaluator:
    """Scores whole populations on a persistent process pool with pre-assigned seeds."""

    def __init__(
        self,
        data_directory: Path,
        epochs: int,
        workers: int = 1,
        seed: int = 0,
    ) -> None:
        """Configures the pool, the epoch budget and the run seed of the evaluator."""
        if workers < 1:
            raise ValueError("Worker count must be at least 1.")
        if epochs < 1:
            raise ValueError("Epoch count must be at least 1.")
        self.epochs = epochs
        self.workers = workers
        self.seed = seed
        self._seed_rng = np.random.default_rng(seed)
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(data_directory,),
        )

    def __call__(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns one fitness value per row, keeping row order regardless of scheduling."""
        seeds = draw_evaluation_seeds(self._seed_rng, len(population))
        tasks = self._build_tasks(population, seeds)
        results = self._executor.map(evaluate_task, tasks, chunksize=1)
        return np.array(list(results), dtype=np.float64)

    def warm_up(self, population: NDArray[np.float64]) -> None:
        """Runs one untimed batch so worker processes start before measurement."""
        seeds = draw_evaluation_seeds(np.random.default_rng(self.seed + 1), len(population))
        tasks = self._build_tasks(population, seeds)
        list(self._executor.map(evaluate_task, tasks, chunksize=1))

    def _build_tasks(
        self,
        population: NDArray[np.float64],
        seeds: NDArray[np.int64],
    ) -> list[FitnessTask]:
        """Converts chromosomes and their assigned seeds into serializable worker tasks."""
        return [
            FitnessTask(
                chromosome=np.array(chromosome, dtype=np.float64),
                epochs=self.epochs,
                seed=int(seeds[index]),
            )
            for index, chromosome in enumerate(population)
        ]

    def close(self) -> None:
        """Shuts down the worker process pool."""
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "UciBatchEvaluator":
        """Enters the runtime context and returns this evaluator."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leaves the runtime context, always shutting down the worker pool."""
        self.close()
