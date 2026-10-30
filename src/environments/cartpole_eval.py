from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import TracebackType

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.networks.numpy_mlp import NumpyMLP

CARTPOLE_ARCHITECTURE: tuple[int, int, int] = (4, 8, 2)


@dataclass(frozen=True)
class EpisodeTrace:
    """States visited, actions taken and rewards received of one played episode."""
    states: NDArray[np.float64]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]


def play_episode(
    select_action: Callable[[NDArray[np.float64]], int],
    seed: int | None = None,
) -> EpisodeTrace:
    """Runs one CartPole episode with the given action function and returns its trace."""
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)
    states: list[NDArray[np.float64]] = []
    actions: list[int] = []
    rewards: list[float] = []
    done = False
    while not done:
        action = int(select_action(state))
        states.append(np.array(state, dtype=np.float64))
        actions.append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        done = terminated or truncated
    env.close()
    return EpisodeTrace(
        states=np.array(states, dtype=np.float64),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float64),
    )


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
        self.seed = seed
        self._seed_rng = np.random.default_rng(seed)
        self._executor = ProcessPoolExecutor(max_workers=workers)

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
    ) -> list[EvaluationTask]:
        """Converts chromosomes and their assigned seeds into serializable worker tasks."""
        return [
            EvaluationTask(
                chromosome=np.array(chromosome, dtype=np.float64),
                architecture=self.architecture,
                episodes=self.episodes,
                seed=int(seeds[index]),
            )
            for index, chromosome in enumerate(population)
        ]

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
