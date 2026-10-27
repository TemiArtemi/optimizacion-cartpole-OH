import numpy as np
from numpy.typing import NDArray


def layer_shapes(architecture: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Returns the weight and bias shapes of a fully connected network."""
    shapes: list[tuple[int, ...]] = []
    for input_size, output_size in zip(architecture[:-1], architecture[1:]):
        shapes.append((input_size, output_size))
        shapes.append((1, output_size))
    return shapes


def flatten_weights(weights: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Concatenates every weight matrix and bias vector into one flat chromosome."""
    return np.concatenate([np.ravel(weight) for weight in weights])


def unflatten_weights(
    chromosome: NDArray[np.float64],
    shapes: list[tuple[int, ...]],
) -> list[NDArray[np.float64]]:
    """Splits a flat chromosome into weight pieces matching the given shapes."""
    expected = int(sum(int(np.prod(shape)) for shape in shapes))
    if len(chromosome) != expected:
        raise ValueError(
            f"Chromosome length {len(chromosome)} does not match architecture length {expected}."
        )
    weights: list[NDArray[np.float64]] = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(np.array(chromosome[offset:offset + size]).reshape(shape))
        offset += size
    return weights


def random_chromosome(
    architecture: tuple[int, ...],
    rng: np.random.Generator,
    scale: float = 0.1,
) -> NDArray[np.float64]:
    """Samples an initial chromosome with scaled normal weights and zero biases."""
    weights: list[NDArray[np.float64]] = []
    for input_size, output_size in zip(architecture[:-1], architecture[1:]):
        weights.append(rng.normal(0.0, scale, size=(input_size, output_size)))
        weights.append(np.zeros((1, output_size)))
    return flatten_weights(weights)


class NumpyMLP:
    """Neural network that decodes a flat chromosome into weight matrices."""

    def __init__(self, architecture: tuple[int, ...], chromosome: NDArray[np.float64]) -> None:
        """Builds the network from a layer size tuple and a flat chromosome."""
        self.architecture = architecture
        self.weights = unflatten_weights(chromosome, layer_shapes(architecture))

    @property
    def chromosome(self) -> NDArray[np.float64]:
        """Returns the current weights encoded as a flat chromosome."""
        return flatten_weights(self.weights)

    def logits(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns the output layer values for one observation."""
        activation = np.asarray(state, dtype=np.float64).ravel()
        layer_count = len(self.weights) // 2
        for layer_index in range(layer_count):
            activation = activation @ self.weights[2 * layer_index]
            activation = activation + np.ravel(self.weights[2 * layer_index + 1])
            if layer_index < layer_count - 1:
                activation = np.maximum(activation, 0.0)
        return activation

    def forward(self, state: NDArray[np.float64]) -> int:
        """Returns the greedy action index for one observation."""
        return int(np.argmax(self.logits(state)))
