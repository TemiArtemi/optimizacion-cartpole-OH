import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from src.networks.torch_mlp import MLPConfig, build_mlp, train_and_evaluate
from src.uci.dataset import load_letter_dataset
from src.uci.preparation import (
    FEATURE_COUNT,
    LETTER_CLASSES,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    build_splits,
    prepare_letter_splits,
    read_letter_table,
    save_splits,
    standardize,
)

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "letter"


def make_test_config() -> MLPConfig:
    """Builds a small valid network configuration for the torch smoke tests."""
    return MLPConfig(
        hidden_sizes=(32,),
        activation="relu",
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=32,
    )


def make_training_sample(
    row_count: int = 400,
    class_count: int = LETTER_CLASSES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a small learnable sample whose features shift with the class index."""
    rng = np.random.default_rng(seed)
    labels = np.arange(row_count, dtype=np.int64) % class_count
    features = rng.normal(size=(row_count, FEATURE_COUNT)) + labels[:, None]
    return features, labels


def make_letter_table(row_count: int = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Builds a synthetic letter table whose columns carry shifted copies of a unique row marker."""
    labels = np.arange(row_count, dtype=np.int64) % LETTER_CLASSES
    row_index = np.arange(row_count, dtype=np.float64)
    features = np.add.outer(row_index, np.arange(FEATURE_COUNT, dtype=np.float64) * row_count)
    return labels, features


def test_read_letter_table_maps_letters_to_indices() -> None:
    """Checks that the parser maps A and Z to 0 and 25 and keeps 16 float features per row."""
    content = "A,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16\nZ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "letter.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("letter-recognition.data", content)
        labels, features = read_letter_table(archive_path)
    assert labels.tolist() == [0, 25]
    assert features.shape == (2, FEATURE_COUNT)
    assert features.dtype == np.float64
    assert features[0, 0] == 1.0


def test_read_letter_table_rejects_unknown_letters() -> None:
    """Checks that a label outside the alphabet is rejected during parsing."""
    content = "a,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16\n"
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "letter.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("letter-recognition.data", content)
        try:
            read_letter_table(archive_path)
            raise AssertionError("expected ValueError for a lowercase letter label")
        except ValueError:
            pass


def test_build_splits_partitions_the_official_blocks() -> None:
    """Checks split sizes and that train and validation exactly cover the official train block."""
    labels, features = make_letter_table()
    train_x, train_y, validation_x, validation_y, test_x, test_y = build_splits(labels, features, seed=0)
    assert train_x.shape == (TRAIN_SIZE, FEATURE_COUNT)
    assert validation_x.shape == (VALIDATION_SIZE, FEATURE_COUNT)
    assert test_x.shape == (TEST_SIZE, FEATURE_COUNT)
    train_markers = set(train_x[:, 0].astype(np.int64).tolist())
    validation_markers = set(validation_x[:, 0].astype(np.int64).tolist())
    test_markers = test_x[:, 0].astype(np.int64)
    assert train_markers.isdisjoint(validation_markers)
    assert train_markers | validation_markers == set(range(TRAIN_SIZE + VALIDATION_SIZE))
    assert np.array_equal(test_markers, np.arange(TRAIN_SIZE + VALIDATION_SIZE, TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE))
    assert np.array_equal(np.sort(np.concatenate([train_y, validation_y])), np.sort(labels[: TRAIN_SIZE + VALIDATION_SIZE]))
    assert np.array_equal(test_y, labels[TRAIN_SIZE + VALIDATION_SIZE :])


def test_build_splits_is_deterministic_per_seed() -> None:
    """Checks that the same seed reproduces the same partitions and a different seed changes them."""
    labels, features = make_letter_table()
    first = build_splits(labels, features, seed=7)
    second = build_splits(labels, features, seed=7)
    third = build_splits(labels, features, seed=8)
    for first_array, second_array in zip(first, second, strict=True):
        assert np.array_equal(first_array, second_array)
    assert not np.array_equal(first[0], third[0])


def test_standardize_centers_and_scales_with_train_statistics() -> None:
    """Checks that the train matrix ends up with zero mean and unit deviation per column."""
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 3.0, size=(1000, FEATURE_COUNT))
    other = rng.normal(-2.0, 7.0, size=(200, FEATURE_COUNT))
    scaled_train, scaled_other = standardize(train, other)
    assert np.allclose(scaled_train.mean(axis=0), 0.0)
    assert np.allclose(scaled_train.std(axis=0), 1.0)
    assert scaled_other.shape == other.shape


def test_prepare_letter_splits_is_deterministic_and_label_preserving() -> None:
    """Checks that preparation is reproducible and keeps the label multiset of every block."""
    labels, features = make_letter_table()
    first = prepare_letter_splits(labels, features, seed=3)
    second = prepare_letter_splits(labels, features, seed=3)
    for first_array, second_array in zip(first, second, strict=True):
        assert np.array_equal(first_array, second_array)
    train_x, train_y, validation_x, validation_y, test_x, test_y = first
    assert np.allclose(train_x.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(train_x.std(axis=0), 1.0, atol=1e-10)
    assert np.array_equal(test_y, labels[TRAIN_SIZE + VALIDATION_SIZE :])
    combined = np.sort(np.concatenate([train_y, validation_y]))
    assert np.array_equal(combined, np.sort(labels[: TRAIN_SIZE + VALIDATION_SIZE]))
    assert validation_x.shape == (VALIDATION_SIZE, FEATURE_COUNT)
    assert test_x.shape == (TEST_SIZE, FEATURE_COUNT)


def test_save_and_load_roundtrip() -> None:
    """Checks that saving and reloading the splits reproduces identical arrays."""
    labels, features = make_letter_table()
    splits = prepare_letter_splits(labels, features, seed=0)
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        save_splits(directory, splits)
        dataset = load_letter_dataset(directory)
    assert np.array_equal(dataset.train.features, splits[0])
    assert np.array_equal(dataset.train.labels, splits[1])
    assert np.array_equal(dataset.validation.features, splits[2])
    assert np.array_equal(dataset.validation.labels, splits[3])
    assert np.array_equal(dataset.test.features, splits[4])
    assert np.array_equal(dataset.test.labels, splits[5])


def test_prepared_repository_data_matches_documented_shapes() -> None:
    """Checks the on-disk letter splits when they have been generated by the prepare script."""
    if not DATA_DIRECTORY.exists():
        return
    dataset = load_letter_dataset(DATA_DIRECTORY)
    assert dataset.train.features.shape == (TRAIN_SIZE, FEATURE_COUNT)
    assert dataset.validation.features.shape == (VALIDATION_SIZE, FEATURE_COUNT)
    assert dataset.test.features.shape == (TEST_SIZE, FEATURE_COUNT)
    assert dataset.train.labels.min() >= 0 and dataset.train.labels.max() < LETTER_CLASSES


def test_build_mlp_output_shape_matches_class_count() -> None:
    """Checks that the built network maps a batch of rows to one logit per class."""
    features, _ = make_training_sample(64)
    model = build_mlp(FEATURE_COUNT, LETTER_CLASSES, make_test_config())
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
    assert logits.shape == (64, LETTER_CLASSES)


def test_build_mlp_rejects_unknown_activation() -> None:
    """Checks that an activation outside the catalog is rejected at build time."""
    config = MLPConfig(
        hidden_sizes=(8,),
        activation="swish",
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=16,
    )
    try:
        build_mlp(FEATURE_COUNT, LETTER_CLASSES, config)
        raise AssertionError("expected ValueError for an unknown activation")
    except ValueError:
        pass


def test_train_and_evaluate_returns_bounded_accuracy() -> None:
    """Checks that a short training run finishes and returns an accuracy in the unit interval."""
    train_x, train_y = make_training_sample(400, seed=0)
    validation_x, validation_y = make_training_sample(100, seed=1)
    accuracy = train_and_evaluate(
        make_test_config(),
        train_x,
        train_y,
        validation_x,
        validation_y,
        class_count=LETTER_CLASSES,
        epochs=2,
        seed=5,
    )
    assert 0.0 <= accuracy <= 1.0


def test_train_and_evaluate_is_deterministic_per_seed() -> None:
    """Checks that repeating a seeded training run reproduces the exact accuracy."""
    train_x, train_y = make_training_sample(200, seed=0)
    validation_x, validation_y = make_training_sample(60, seed=1)
    first = train_and_evaluate(
        make_test_config(),
        train_x,
        train_y,
        validation_x,
        validation_y,
        class_count=LETTER_CLASSES,
        epochs=2,
        seed=9,
    )
    second = train_and_evaluate(
        make_test_config(),
        train_x,
        train_y,
        validation_x,
        validation_y,
        class_count=LETTER_CLASSES,
        epochs=2,
        seed=9,
    )
    assert first == second


def main() -> None:
    """Runs every phase 3 test and reports the result."""
    test_read_letter_table_maps_letters_to_indices()
    test_read_letter_table_rejects_unknown_letters()
    test_build_splits_partitions_the_official_blocks()
    test_build_splits_is_deterministic_per_seed()
    test_standardize_centers_and_scales_with_train_statistics()
    test_prepare_letter_splits_is_deterministic_and_label_preserving()
    test_save_and_load_roundtrip()
    test_prepared_repository_data_matches_documented_shapes()
    test_build_mlp_output_shape_matches_class_count()
    test_build_mlp_rejects_unknown_activation()
    test_train_and_evaluate_returns_bounded_accuracy()
    test_train_and_evaluate_is_deterministic_per_seed()
    print("All phase 3 tests passed.")


if __name__ == "__main__":
    main()
