from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DatasetSplit:
    """Feature matrix and label vector of one dataset partition."""
    features: NDArray[np.float64]
    labels: NDArray[np.int64]


@dataclass(frozen=True)
class LetterDataset:
    """Train, validation and test partitions of the letter recognition dataset."""
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


def load_split(path: Path) -> DatasetSplit:
    """Loads one npz partition file into a typed dataset split."""
    with np.load(path) as archive:
        features = np.asarray(archive["features"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    return DatasetSplit(features=features, labels=labels)


def load_letter_dataset(root: Path) -> LetterDataset:
    """Loads the three prepared letter partitions stored under the given directory."""
    return LetterDataset(
        train=load_split(root / "train.npz"),
        validation=load_split(root / "val.npz"),
        test=load_split(root / "test.npz"),
    )
