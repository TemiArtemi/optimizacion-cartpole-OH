from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def mean_std(values: NDArray[np.float64]) -> tuple[float, float]:
    """Returns the mean and the sample standard deviation of the given values."""
    array = np.asarray(values, dtype=np.float64)
    deviation = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), deviation


def mean_std_curve(
    curves: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns the pointwise mean and standard deviation across several curves."""
    stacked = np.stack(curves)
    return stacked.mean(axis=0), stacked.std(axis=0)


def success_rate(targets: Sequence[int | None]) -> float:
    """Returns the fraction of runs that reached the target score."""
    if not targets:
        return 0.0
    return sum(value is not None for value in targets) / len(targets)


def evaluations_to_target(
    best_curve: NDArray[np.float64],
    evaluations: NDArray[np.int64],
    target: float,
) -> int | None:
    """Returns the evaluations at which the curve first reached the target, if ever."""
    reached = np.nonzero(best_curve >= target)[0]
    if reached.size == 0:
        return None
    return int(evaluations[reached[0]])
