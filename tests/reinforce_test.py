import numpy as np

from src.baselines.reinforce import (
    action_probabilities,
    policy_gradients,
    run_reinforce,
    softmax,
)
from src.environments.cartpole_eval import play_episode
from src.networks.numpy_mlp import (
    NumpyMLP,
    flatten_weights,
    random_chromosome,
)


def test_softmax_normalizes_and_is_numerically_stable() -> None:
    """Checks that softmax returns a valid distribution even for extreme logits."""
    probabilities = softmax(np.array([1000.0, 1001.0]))
    assert np.isclose(float(probabilities.sum()), 1.0)
    assert np.all(probabilities > 0.0)
    dominant = softmax(np.array([500.0, -500.0]))
    assert np.isclose(float(dominant[0]), 1.0)
    assert not np.any(np.isnan(dominant))


def test_play_episode_returns_a_seeded_trace() -> None:
    """Checks that a fixed seed replays an episode and that rewards build unit returns."""
    def select_zero(state: np.ndarray) -> int:
        """Always picks the first action as a deterministic test policy."""
        return 0

    first = play_episode(select_zero, seed=123)
    second = play_episode(select_zero, seed=123)
    assert first.states.shape[1] == 4
    assert len(first.actions) == len(first.rewards)
    assert np.all(first.rewards == 1.0)
    returns_to_go = np.cumsum(first.rewards[::-1])[::-1]
    assert np.all(np.diff(returns_to_go) == -1.0)
    assert returns_to_go[-1] == 1.0
    assert len(first.actions) == len(second.actions)
    assert np.array_equal(first.actions, second.actions)


def test_policy_gradients_match_finite_differences() -> None:
    """Checks the hand written backpropagation against numerical derivatives of the objective."""
    rng = np.random.default_rng(0)
    architecture = (2, 3, 2)
    chromosome = random_chromosome(architecture, rng, scale=1.0)
    network = NumpyMLP(architecture, chromosome)
    states = rng.normal(size=(3, 2))
    actions = np.array([0, 1, 1], dtype=np.int64)
    coefficients = np.array([2.0, -1.5, 0.5])
    weight_gradients, bias_gradients = policy_gradients(network, states, actions, coefficients)
    gradient_list = [part for pair in zip(weight_gradients, bias_gradients) for part in pair]
    analytic = flatten_weights(gradient_list)
    assert np.any(np.abs(analytic) > 0.0)

    def objective(candidate: np.ndarray) -> float:
        """Returns the coefficient weighted log likelihood of the sampled actions."""
        policy = NumpyMLP(architecture, candidate)
        total = 0.0
        for state, action, coefficient in zip(states, actions, coefficients):
            probabilities = action_probabilities(policy, state)
            total += float(coefficient) * float(np.log(probabilities[action]))
        return total

    epsilon = 1e-6
    numeric = np.empty(len(chromosome))
    for index in range(len(chromosome)):
        perturbed = chromosome.copy()
        perturbed[index] += epsilon
        upper = objective(perturbed)
        perturbed[index] -= 2.0 * epsilon
        lower = objective(perturbed)
        numeric[index] = (upper - lower) / (2.0 * epsilon)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_run_reinforce_is_reproducible_and_sized() -> None:
    """Checks that a fixed seed reproduces the full return history and final weights."""
    arguments = {"architecture": (4, 8, 2), "episodes": 30, "batch_size": 10, "seed": 7}
    first = run_reinforce(**arguments)
    second = run_reinforce(**arguments)
    assert first.returns.shape == (30,)
    assert np.array_equal(first.returns, second.returns)
    assert np.array_equal(first.final_chromosome, second.final_chromosome)
    curve = first.best_curve
    assert curve.shape == (30,)
    assert np.all(np.diff(curve) >= 0.0)
    assert curve[-1] == float(np.max(first.returns))


def test_zero_learning_rate_keeps_the_initial_weights() -> None:
    """Checks that a zero learning rate leaves the network exactly at its initialization."""
    expected_initial = random_chromosome((4, 8, 2), np.random.default_rng(9))
    result = run_reinforce((4, 8, 2), episodes=5, batch_size=5, seed=9, learning_rate=0.0)
    assert np.array_equal(result.final_chromosome, expected_initial)


def test_run_reinforce_rejects_invalid_configuration() -> None:
    """Checks that non-positive budgets and negative learning rates are rejected."""
    try:
        run_reinforce((4, 8, 2), episodes=0)
        raise AssertionError("expected ValueError for zero episodes")
    except ValueError:
        pass
    try:
        run_reinforce((4, 8, 2), episodes=5, batch_size=0)
        raise AssertionError("expected ValueError for zero batch size")
    except ValueError:
        pass
    try:
        run_reinforce((4, 8, 2), episodes=5, learning_rate=-0.1)
        raise AssertionError("expected ValueError for negative learning rate")
    except ValueError:
        pass


def main() -> None:
    """Runs every reinforce test and reports the result."""
    test_softmax_normalizes_and_is_numerically_stable()
    test_play_episode_returns_a_seeded_trace()
    test_policy_gradients_match_finite_differences()
    test_run_reinforce_is_reproducible_and_sized()
    test_zero_learning_rate_keeps_the_initial_weights()
    test_run_reinforce_rejects_invalid_configuration()
    print("All reinforce tests passed.")


if __name__ == "__main__":
    main()
