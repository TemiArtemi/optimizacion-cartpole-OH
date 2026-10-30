import numpy as np

from src.utils.metrics import evaluations_to_target, mean_std, mean_std_curve, success_rate


def test_mean_std_uses_sample_deviation() -> None:
    """Checks that the reported deviation corresponds to the sample formula with n minus one."""
    mean, deviation = mean_std(np.array([1.0, 2.0, 3.0, 4.0]))
    assert mean == 2.5
    assert np.isclose(deviation, np.sqrt(5.0 / 3.0))


def test_mean_std_handles_a_single_value() -> None:
    """Checks that a single run reports zero deviation instead of an undefined one."""
    mean, deviation = mean_std(np.array([7.0]))
    assert mean == 7.0
    assert deviation == 0.0


def test_mean_std_curve_averages_runs_pointwise() -> None:
    """Checks that curves of several runs are combined generation by generation."""
    mean_curve, std_curve = mean_std_curve(
        [np.array([1.0, 3.0]), np.array([3.0, 5.0])]
    )
    assert np.allclose(mean_curve, [2.0, 4.0])
    assert np.allclose(std_curve, [1.0, 1.0])


def test_success_rate_counts_reached_targets() -> None:
    """Checks that the success rate counts only the runs that reached the target."""
    assert success_rate([100, None, 200, None, 50]) == 0.6
    assert success_rate([None, None]) == 0.0
    assert success_rate([]) == 0.0


def test_evaluations_to_target_returns_first_crossing() -> None:
    """Checks that the target metric reports the first crossing and handles a missing one."""
    best_curve = np.array([10.0, 20.0, 500.0, 500.0])
    evaluations = np.array([50, 100, 150, 200])
    assert evaluations_to_target(best_curve, evaluations, 500.0) == 150
    assert evaluations_to_target(best_curve, evaluations, 15.0) == 100
    assert evaluations_to_target(best_curve, evaluations, 5.0) == 50
    assert evaluations_to_target(best_curve, evaluations, 999.0) is None


def main() -> None:
    """Runs every metric test and reports the result."""
    test_mean_std_uses_sample_deviation()
    test_mean_std_handles_a_single_value()
    test_mean_std_curve_averages_runs_pointwise()
    test_success_rate_counts_reached_targets()
    test_evaluations_to_target_returns_first_crossing()
    print("All metric tests passed.")


if __name__ == "__main__":
    main()
