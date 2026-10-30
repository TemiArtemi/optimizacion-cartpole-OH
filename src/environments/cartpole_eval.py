from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import TracebackType

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.networks.numpy_mlp import NumpyMLP

CARTPOLE_ARCHITECTURE: tuple[int, int, int] = (4, 8, 2)


def evaluate_chromosome(
    chromosome: NDArray[np.float64],
    architecture: tuple[int, ...] = CARTPOLE_ARCHITECTURE,
    episodes: int = 1,
    seed: int | None = None,
    render: bool = False,
) -> float:
    """Evaluates a chromosome in CartPole and returns the mean reward per episode."""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    network = NumpyMLP(architecture, chromosome)
    rewards: list[float] = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total = 0.0
        done = False
        while not done:
            action = network.forward(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


@dataclass(frozen=True)
class EvaluationTask:
    """Bundles one chromosome with its evaluation settings for a worker process."""
    chromosome: NDArray[np.float64]
    architecture: tuple[int, ...]
    episodes: int
    seed: int


def draw_evaluation_seeds(rng: np.random.Generator, count: int) -> NDArray[np.int64]:
    """Draws one evaluation seed per individual before any task is dispatched."""
    return rng.integers(1_000_000, size=count).astype(np.int64)


def evaluate_task(task: EvaluationTask) -> float:
    """Evaluates one task inside a worker process and returns the mean reward."""
    return evaluate_chromosome(
        task.chromosome,
        architecture=task.architecture,
        episodes=task.episodes,
        seed=task.seed,
    )


class CartpoleBatchEvaluator:
    """Scores whole populations on a persistent process pool with pre-assigned seeds."""

    def __init__(
        self,
        architecture: tuple[int, ...] = CARTPOLE_ARCHITECTURE,
        episodes: int = 1,
        workers: int = 1,
        seed: int = 0,
    ) -> None:
        """Configures the pool, the episode count and the run seed of the evaluator."""
        if workers < 1:
            raise ValueError("Worker count must be at least 1.")
        if episodes < 1:
            raise ValueError("Episode count must be at least 1.")
        self.architecture = architecture
        self.episodes = episodes
        self.workers = workers
        self._seed_rng = np.random.default_rng(seed)
        self._executor = ProcessPoolExecutor(max_workers=workers)

    def __call__(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns one fitness value per row, keeping row order regardless of scheduling."""
        seeds = draw_evaluation_seeds(self._seed_rng, len(population))
        tasks = [
            EvaluationTask(
                chromosome=np.array(chromosome, dtype=np.float64),
                architecture=self.architecture,
                episodes=self.episodes,
                seed=int(seeds[index]),
            )
            for index, chromosome in enumerate(population)
        ]
        results = self._executor.map(evaluate_task, tasks, chunksize=1)
        return np.array(list(results), dtype=np.float64)

    def close(self) -> None:
        """Shuts down the worker process pool."""
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "CartpoleBatchEvaluator":
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
