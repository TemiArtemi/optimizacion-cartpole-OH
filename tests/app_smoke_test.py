import numpy as np

from app.cartpole_view import (
    build_animation,
    list_controllers,
    load_controller,
    play_controller_episode,
    save_controller,
    train_genetic,
    train_random,
    train_reinforce,
)
from app.letter_view import build_chromosome, config_summary
from src.genetic.hyperparams_chromosome import DEFAULT_CONFIG, decode
from src.networks.numpy_mlp import random_chromosome
from src.environments.cartpole_eval import CARTPOLE_ARCHITECTURE


def test_save_load_controller_roundtrip() -> None:
    """Checks that a saved controller file reloads its chromosome and metadata exactly."""
    rng = np.random.default_rng(0)
    chromosome = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    path = save_controller(chromosome, "Algoritmo genético (torneo k=3)", 4242, 487.5)
    try:
        assert path.name in list_controllers()
        loaded, metadata = load_controller(path.name)
        assert np.array_equal(loaded, chromosome)
        assert metadata["fitness"] == 487.5
        assert metadata["method"] == "Algoritmo genético (torneo k=3)"
    finally:
        path.unlink(missing_ok=True)


def test_build_animation_produces_one_frame_per_step() -> None:
    """Checks that the animation contains one frame per recorded state with both traces."""
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.05, 0.0],
            [0.2, 0.0, 0.10, 0.0],
        ]
    )
    figure = build_animation(states, "prueba")
    assert len(figure.frames) == 3
    assert len(figure.data) == 3
    assert len(figure.frames[1].data) == 3


def test_play_controller_episode_returns_states_and_score() -> None:
    """Checks that one played episode records states and a matching score."""
    rng = np.random.default_rng(1)
    chromosome = random_chromosome(CARTPOLE_ARCHITECTURE, rng)
    states, score = play_controller_episode(chromosome, 42)
    assert states.ndim == 2
    assert states.shape[1] == 4
    assert states.shape[0] == score
    assert score >= 1


def test_training_functions_run_with_a_tiny_budget() -> None:
    """Checks that the three training entry points of the tab produce usable results."""
    genetic, genetic_seconds = train_genetic(100, 0, lambda stats: None)
    assert len(genetic.history) == 2
    assert genetic_seconds > 0.0
    random_result, random_seconds = train_random(100, 0)
    assert len(random_result.history) == 2
    assert random_seconds > 0.0
    reinforce, reinforce_seconds = train_reinforce(100, 0)
    assert len(reinforce.returns) == 300
    assert reinforce_seconds > 0.0


def test_build_chromosome_decodes_to_chosen_configuration() -> None:
    """Checks that the interface values become the exact decoded network configuration."""
    chromosome = build_chromosome(2, (6, 5, 4, 4), 1, -3, -4, 0.2, 6)
    config = decode(chromosome)
    assert config.hidden_sizes == (64, 32)
    assert config.activation == "tanh"
    assert np.isclose(config.learning_rate, 1e-3)
    assert np.isclose(config.weight_decay, 1e-4)
    assert config.dropout == 0.2
    assert config.batch_size == 64


def test_config_summary_reports_spanish_labels() -> None:
    """Checks that the configuration table shows readable Spanish labels."""
    summary = config_summary(DEFAULT_CONFIG)
    assert summary["Capas ocultas"] == 2
    assert summary["Anchos"] == "64 / 64"
    assert summary["Activación"] == "relu"
    assert summary["Tamaño de lote"] == 64


def main() -> None:
    """Runs every app smoke test and reports the result."""
    test_save_load_controller_roundtrip()
    test_build_animation_produces_one_frame_per_step()
    test_play_controller_episode_returns_states_and_score()
    test_training_functions_run_with_a_tiny_budget()
    test_build_chromosome_decodes_to_chosen_configuration()
    test_config_summary_reports_spanish_labels()
    print("All app smoke tests passed.")


if __name__ == "__main__":
    main()
