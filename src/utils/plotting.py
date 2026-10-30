from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_convergence(
    curves: dict[str, NDArray[np.float64]],
    evaluations: NDArray[np.int64],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    colors: tuple[str, ...] = ("tab:blue", "tab:orange", "tab:green", "tab:red"),
) -> None:
    """Saves a convergence figure with mean curves and one standard deviation bands."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for index, (label, matrix) in enumerate(curves.items()):
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0)
        color = colors[index % len(colors)]
        axis.plot(evaluations, mean_curve, label=label, color=color)
        axis.fill_between(
            evaluations,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.2,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
