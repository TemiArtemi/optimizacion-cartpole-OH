import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

UCI_URL = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
ARCHIVE_MEMBER = "letter-recognition.data"
SEED = 0
TRAIN_SIZE = 12_000
VALIDATION_SIZE = 4_000
TEST_SIZE = 4_000
LETTER_CLASSES = 26
FEATURE_COUNT = 16
SplitBundle = tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]


def download_archive(url: str, destination: Path) -> None:
    """Downloads the UCI letter archive into the destination path when it is missing."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def read_letter_table(archive_path: Path) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Extracts integer class labels and the float feature matrix from the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read(ARCHIVE_MEMBER).decode("ascii")
    labels: list[int] = []
    rows: list[list[float]] = []
    for line in text.strip().splitlines():
        fields = line.split(",")
        labels.append(ord(fields[0]) - ord("A"))
        rows.append([float(field) for field in fields[1:]])
    labels_array = np.array(labels, dtype=np.int64)
    features_array = np.array(rows, dtype=np.float64)
    if features_array.ndim != 2 or features_array.shape[1] != FEATURE_COUNT:
        raise ValueError("Every letter row must contain exactly 16 features.")
    if labels_array.size and (labels_array.min() < 0 or labels_array.max() >= LETTER_CLASSES):
        raise ValueError("Letter labels must lie between 0 and 25.")
    return labels_array, features_array


def build_splits(
    labels: NDArray[np.int64],
    features: NDArray[np.float64],
    seed: int = SEED,
) -> SplitBundle:
    """Shuffles the official train block into train and validation and keeps the test block intact."""
    expected_rows = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
    if labels.shape[0] != expected_rows or features.shape[0] != expected_rows:
        raise ValueError(f"The letter table must contain exactly {expected_rows} rows.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(TRAIN_SIZE + VALIDATION_SIZE)
    train_index = order[:TRAIN_SIZE]
    validation_index = order[TRAIN_SIZE:]
    test_index = np.arange(TRAIN_SIZE + VALIDATION_SIZE, expected_rows)
    return (
        features[train_index],
        labels[train_index],
        features[validation_index],
        labels[validation_index],
        features[test_index],
        labels[test_index],
    )


def standardize(
    train_features: NDArray[np.float64],
    *other_features: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Rescales every feature matrix with the mean and deviation of the train matrix."""
    mean = train_features.mean(axis=0)
    deviation = train_features.std(axis=0)
    deviation = np.where(deviation == 0.0, 1.0, deviation)
    return tuple((matrix - mean) / deviation for matrix in (train_features, *other_features))


def prepare_letter_splits(
    labels: NDArray[np.int64],
    features: NDArray[np.float64],
    seed: int = SEED,
) -> SplitBundle:
    """Builds the three partitions and standardizes them with the train statistics."""
    train_x, train_y, validation_x, validation_y, test_x, test_y = build_splits(labels, features, seed)
    train_x, validation_x, test_x = standardize(train_x, validation_x, test_x)
    return train_x, train_y, validation_x, validation_y, test_x, test_y


def save_splits(directory: Path, splits: SplitBundle) -> None:
    """Writes the train, validation and test partitions into compressed npz files."""
    directory.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "test")
    feature_matrices = splits[0::2]
    label_vectors = splits[1::2]
    for name, features, labels in zip(names, feature_matrices, label_vectors, strict=True):
        np.savez_compressed(directory / f"{name}.npz", features=features, labels=labels)
