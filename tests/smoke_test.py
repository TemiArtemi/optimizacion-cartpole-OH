import numpy as np

from src.environments.cartpole_eval import CARTPOLE_ARCHITECTURE, evaluate_chromosome
from src.genetic.operators import gaussian_mutation, one_point_crossover
from src.networks.numpy_mlp import (
    NumpyMLP,
    layer_shapes,
    random_chromosome,
    unflatten_weights,
)


def test_chromosome_roundtrip() -> None:
    """Checks that decoding and re-encoding a chromosome reproduces the same weights."""
    rng = np.random.default_rng(0)
    chromosome = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    network = NumpyMLP(CARTPOLE_ARCHITECTURE, chromosome)
    assert np.array_equal(network.chromosome, chromosome)
    assert network.chromosome.shape == (58,)


def test_shape_validation() -> None:
    """Checks that a chromosome whose length mismatches the architecture is rejected."""
    try:
        unflatten_weights(np.zeros(10), layer_shapes((4, 8, 2)))
        raise AssertionError("expected ValueError for mismatched chromosome length")
    except ValueError:
        pass


def test_forward_returns_valid_action() -> None:
    """Checks that the network always returns a valid CartPole action."""
    rng = np.random.default_rng(2)
    network = NumpyMLP(CARTPOLE_ARCHITECTURE, random_chromosome(CARTPOLE_ARCHITECTURE, rng))
    for _ in range(10):
        state = rng.uniform(-1.0, 1.0, size=4)
        action = network.forward(state)
        assert isinstance(action, int)
        assert action in (0, 1)


def test_crossover_is_pure_and_recombines_parents() -> None:
    """Checks that crossover returns children built from both parents without side effects."""
    rng = np.random.default_rng(3)
    parent_a = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    parent_b = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    parent_a_copy = parent_a.copy()
    parent_b_copy = parent_b.copy()
    child_a, child_b = one_point_crossover(parent_a, parent_b, rng)
    assert child_a.shape == parent_a.shape
    assert child_b.shape == parent_b.shape
    from_a = np.isclose(child_a, parent_a)
    from_b = np.isclose(child_a, parent_b)
    assert np.all(from_a | from_b)
    assert np.any(from_a) and np.any(from_b)
    assert np.array_equal(parent_a, parent_a_copy)
    assert np.array_equal(parent_b, parent_b_copy)


def test_mutation_is_pure_and_respects_rate() -> None:
    """Checks that mutation copies the chromosome and only alters genes at the given rate."""
    rng = np.random.default_rng(4)
    chromosome = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    original = chromosome.copy()
    unchanged = gaussian_mutation(chromosome, rng, rate=0.0)
    assert np.array_equal(unchanged, original)
    changed = gaussian_mutation(chromosome, rng, rate=1.0, sigma=0.5)
    assert not np.allclose(changed, original)
    assert np.array_equal(chromosome, original)


def test_evaluation_is_seeded_and_bounded() -> None:
    """Checks that seeded evaluations are reproducible and rewards stay within CartPole bounds."""
    rng = np.random.default_rng(5)
    chromosome = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    first = evaluate_chromosome(chromosome, episodes=3, seed=42)
    second = evaluate_chromosome(chromosome, episodes=3, seed=42)
    assert first == second
    assert 0.0 < first <= 500.0


def main() -> None:
    """Runs every phase 0 smoke test and reports the result."""
    test_chromosome_roundtrip()
    test_shape_validation()
    test_forward_returns_valid_action()
    test_crossover_is_pure_and_recombines_parents()
    test_mutation_is_pure_and_respects_rate()
    test_evaluation_is_seeded_and_bounded()
    print("All phase 0 smoke tests passed.")


if __name__ == "__main__":
    main()
