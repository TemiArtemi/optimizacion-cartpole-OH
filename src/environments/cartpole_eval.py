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
