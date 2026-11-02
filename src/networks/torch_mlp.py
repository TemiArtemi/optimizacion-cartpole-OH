from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "leaky_relu")


@dataclass(frozen=True)
class MLPConfig:
    """Architecture and training hyperparameters of a tabular PyTorch MLP."""
    hidden_sizes: tuple[int, ...]
    activation: str
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int


def activation_module(name: str) -> nn.Module:
    """Returns the activation module registered under the given catalog name."""
    modules = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
    if name not in modules:
        raise ValueError(f"Unknown activation '{name}'.")
    return modules[name]()


def build_mlp(input_size: int, output_size: int, config: MLPConfig) -> nn.Module:
    """Builds a fully connected network whose hidden layers follow the given configuration."""
    layers: list[nn.Module] = []
    previous_size = input_size
    for hidden_size in config.hidden_sizes:
        layers.append(nn.Linear(previous_size, hidden_size))
        layers.append(activation_module(config.activation))
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        previous_size = hidden_size
    layers.append(nn.Linear(previous_size, output_size))
    return nn.Sequential(*layers)


def train_and_evaluate(
    config: MLPConfig,
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    evaluation_features: NDArray[np.float64],
    evaluation_labels: NDArray[np.int64],
    class_count: int,
    epochs: int,
    seed: int,
) -> float:
    """Trains the configured network and returns the evaluation accuracy using single-threaded math."""
    if epochs < 1:
        raise ValueError("Epoch count must be at least 1.")
    if class_count < 2:
        raise ValueError("Class count must be at least 2.")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.int64)
    evaluation_x = torch.tensor(evaluation_features, dtype=torch.float32)
    evaluation_y = torch.tensor(evaluation_labels, dtype=torch.int64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = build_mlp(train_x.shape[1], class_count, config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(evaluation_x).argmax(dim=1)
        accuracy = (predictions == evaluation_y).float().mean()
    return float(accuracy)
