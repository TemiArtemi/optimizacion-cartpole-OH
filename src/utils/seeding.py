import numpy as np
from numpy.typing import NDArray


def draw_evaluation_seeds(rng: np.random.Generator, count: int) -> NDArray[np.int64]:
    """Draws one evaluation seed per individual before any task is dispatched."""
    return rng.integers(1_000_000, size=count).astype(np.int64)
