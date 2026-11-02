import numpy as np
from numpy.typing import NDArray

from src.networks.torch_mlp import ACTIVATIONS, MLPConfig

CHROMOSOME_LENGTH = 10
LAYER_COUNT_GENE = 0
WIDTH_GENES = (1, 2, 3, 4)
ACTIVATION_GENE = 5
LEARNING_RATE_GENE = 6
WEIGHT_DECAY_GENE = 7
DROPOUT_GENE = 8
BATCH_SIZE_GENE = 9
DISCRETE_GENES = (LAYER_COUNT_GENE, *WIDTH_GENES, BATCH_SIZE_GENE)
GENE_LOW = np.array([1.0, 4.0, 4.0, 4.0, 4.0, 0.0, -4.0, -6.0, 0.0, 4.0], dtype=np.float64)
GENE_HIGH = np.array([4.0, 8.0, 8.0, 8.0, 8.0, 2.0, -1.0, -2.0, 0.5, 8.0], dtype=np.float64)
MUTATION_SIGMA = 0.3

DEFAULT_CONFIG = MLPConfig(
    hidden_sizes=(64, 64),
    activation="relu",
    dropout=0.2,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=64,
)


def clip_chromosome(chromosome: NDArray[np.float64]) -> NDArray[np.float64]:
    """Returns a copy of the chromosome with every gene clamped to its allowed bounds."""
    array = np.asarray(chromosome, dtype=np.float64)
    if array.shape != (CHROMOSOME_LENGTH,):
        raise ValueError(f"Chromosome must contain exactly {CHROMOSOME_LENGTH} genes.")
    return np.clip(array, GENE_LOW, GENE_HIGH)


def decode(chromosome: NDArray[np.float64]) -> MLPConfig:
    """Converts a mixed chromosome into the network configuration it encodes."""
    genes = clip_chromosome(chromosome)
    layer_count = int(round(genes[LAYER_COUNT_GENE]))
    hidden_sizes = tuple(2 ** int(round(genes[index])) for index in WIDTH_GENES[:layer_count])
    activation = ACTIVATIONS[int(round(genes[ACTIVATION_GENE]))]
    return MLPConfig(
        hidden_sizes=hidden_sizes,
        activation=activation,
        dropout=float(genes[DROPOUT_GENE]),
        learning_rate=float(10.0 ** genes[LEARNING_RATE_GENE]),
        weight_decay=float(10.0 ** genes[WEIGHT_DECAY_GENE]),
        batch_size=2 ** int(round(genes[BATCH_SIZE_GENE])),
    )


def random_chromosome(rng: np.random.Generator) -> NDArray[np.float64]:
    """Samples one random mixed chromosome within the allowed gene bounds."""
    return rng.uniform(GENE_LOW, GENE_HIGH)


def mutate_gene(
    index: int,
    value: float,
    rng: np.random.Generator,
    sigma: float = MUTATION_SIGMA,
) -> float:
    """Returns a type-appropriate mutation of the gene stored at the given index."""
    if index == ACTIVATION_GENE:
        current = int(np.clip(round(value), GENE_LOW[index], GENE_HIGH[index]))
        offset = int(rng.integers(1, len(ACTIVATIONS)))
        return float((current + offset) % len(ACTIVATIONS))
    if index in DISCRETE_GENES:
        stepped = round(value + rng.normal(0.0, sigma))
        return float(np.clip(stepped, GENE_LOW[index], GENE_HIGH[index]))
    return float(np.clip(value + rng.normal(0.0, sigma), GENE_LOW[index], GENE_HIGH[index]))
