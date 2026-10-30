from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.environments.cartpole_eval import play_episode
from src.networks.numpy_mlp import NumpyMLP, random_chromosome


@dataclass(frozen=True)
class ReinforceResult:
    """Per-episode returns and final policy weights of one training run."""
    returns: NDArray[np.float64]
    final_chromosome: NDArray[np.float64]

    @property
    def best_curve(self) -> NDArray[np.float64]:
        """Returns the best return seen after each episode of the run."""
        return np.maximum.accumulate(self.returns)


def softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Converts logits into a numerically stable probability distribution."""
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


def action_probabilities(
    network: NumpyMLP,
    state: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns the softmax action probabilities of the policy for one state."""
    return softmax(network.logits(state))


def sample_action(
    network: NumpyMLP,
    state: NDArray[np.float64],
    rng: np.random.Generator,
) -> int:
    """Samples one action from the policy probabilities for one state."""
    probabilities = action_probabilities(network, state)
    return int(rng.choice(len(probabilities), p=probabilities))


def forward_with_cache(
    network: NumpyMLP,
    state: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Runs the forward pass keeping layer inputs and pre-activations for the backward pass."""
    layer_inputs: list[NDArray[np.float64]] = []
    preactivations: list[NDArray[np.float64]] = []
    activation = np.asarray(state, dtype=np.float64).ravel()
    layer_count = len(network.weights) // 2
    for layer_index in range(layer_count):
        weight = network.weights[2 * layer_index]
        bias = network.weights[2 * layer_index + 1]
        layer_inputs.append(activation)
        preactivation = activation @ weight + np.ravel(bias)
        preactivations.append(preactivation)
        if layer_index < layer_count - 1:
            activation = np.maximum(preactivation, 0.0)
        else:
            activation = preactivation
    return layer_inputs, preactivations


def policy_gradients(
    network: NumpyMLP,
    states: NDArray[np.float64],
    actions: NDArray[np.int64],
    coefficients: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Accumulates the policy likelihood gradients scaled by one coefficient per step."""
    layer_count = len(network.weights) // 2
    weight_gradients = [np.zeros_like(network.weights[2 * index]) for index in range(layer_count)]
    bias_gradients = [np.zeros_like(network.weights[2 * index + 1]) for index in range(layer_count)]
    for state, action, coefficient in zip(states, actions, coefficients):
        layer_inputs, preactivations = forward_with_cache(network, state)
        probabilities = softmax(preactivations[-1])
        one_hot = np.zeros_like(probabilities)
        one_hot[action] = 1.0
        delta = float(coefficient) * (one_hot - probabilities)
        for layer_index in reversed(range(layer_count)):
            weight_gradients[layer_index] += np.outer(layer_inputs[layer_index], delta)
            bias_gradients[layer_index] += delta
            if layer_index > 0:
                hidden_mask = preactivations[layer_index - 1] > 0.0
                delta = (delta @ network.weights[2 * layer_index].T) * hidden_mask
    return weight_gradients, bias_gradients


def run_reinforce(
    architecture: tuple[int, ...],
    episodes: int,
    learning_rate: float = 0.001,
    batch_size: int = 10,
    seed: int | None = None,
) -> ReinforceResult:
    """Trains a policy with REINFORCE over a fixed episode budget and returns its history."""
    if episodes < 1:
        raise ValueError("Episode count must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if learning_rate < 0.0:
        raise ValueError("Learning rate must not be negative.")
    rng = np.random.default_rng(seed)
    network = NumpyMLP(architecture, random_chromosome(architecture, rng))
    returns = np.empty(episodes, dtype=np.float64)
    episode_index = 0

    def select_action(state: NDArray[np.float64]) -> int:
        """Samples one policy action for the running training episode."""
        return sample_action(network, state, rng)

    while episode_index < episodes:
        batch_count = min(batch_size, episodes - episode_index)
        batch_states: list[NDArray[np.float64]] = []
        batch_actions: list[NDArray[np.int64]] = []
        batch_rewards: list[NDArray[np.float64]] = []
        for _ in range(batch_count):
            episode_seed = int(rng.integers(1_000_000))
            trace = play_episode(select_action, seed=episode_seed)
            total_return = float(np.sum(trace.rewards))
            returns[episode_index] = total_return
            episode_index += 1
            batch_states.append(trace.states)
            batch_actions.append(trace.actions)
            batch_rewards.append(trace.rewards)
        baseline = float(np.mean([float(np.sum(rewards)) for rewards in batch_rewards]))
        states_array = np.concatenate(batch_states)
        actions_array = np.concatenate(batch_actions)
        coefficients = np.concatenate(
            [np.cumsum(rewards[::-1])[::-1] - baseline for rewards in batch_rewards]
        )
        weight_gradients, bias_gradients = policy_gradients(
            network, states_array, actions_array, coefficients
        )
        scale = learning_rate / batch_count
        for layer_index in range(len(weight_gradients)):
            network.weights[2 * layer_index] += scale * weight_gradients[layer_index]
            network.weights[2 * layer_index + 1] += scale * bias_gradients[layer_index]
    return ReinforceResult(returns=returns, final_chromosome=network.chromosome)
