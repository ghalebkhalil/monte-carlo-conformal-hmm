from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

class UndefinedHMMEstimateError(ValueError):
    """
    Raised when the empirical-frequency HMM estimator is undefined
    because a required denominator is zero.
    """
    pass


def _validate_probability_vector(
    probabilities: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    """
    Validate and return a one-dimensional probability vector.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if probabilities.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if np.any(probabilities < 0):
        raise ValueError(f"{name} cannot contain negative probabilities.")

    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(
            f"{name} must sum to 1. "
            f"Current sum is {probabilities.sum():.12f}."
        )

    return probabilities


def _validate_stochastic_matrix(
    matrix: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    """
    Validate and return a row-stochastic matrix.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty.")

    if np.any(matrix < 0):
        raise ValueError(f"{name} cannot contain negative probabilities.")

    row_sums = matrix.sum(axis=1)

    if not np.allclose(row_sums, 1.0):
        raise ValueError(
            f"Every row of {name} must sum to 1. "
            f"Current row sums are {row_sums}."
        )

    return matrix


def simulate_hmm(
    initial_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    observation_matrix: ArrayLike,
    length: int,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Simulate a discrete Hidden Markov Model.

    Parameters
    ----------
    initial_distribution
        Initial state probabilities.

        If there are K hidden states, this has length K.

    transition_matrix
        K x K matrix P, where

            P[i, j] = P(X_{t+1} = j | X_t = i).

    observation_matrix
        K x M matrix B, where

            B[i, y] = P(Y_t = y | X_t = i).

        Here M is the number of possible observations.

    length
        Number of time points to simulate.

    rng
        NumPy random-number generator. If None, a new generator is
        constructed.

    Returns
    -------
    states
        Integer array of hidden states X_1, ..., X_length.

    observations
        Integer array of observations Y_1, ..., Y_length.
    """
    if not isinstance(length, (int, np.integer)):
        raise TypeError("length must be an integer.")

    if length <= 0:
        raise ValueError("length must be positive.")

    initial_distribution = _validate_probability_vector(
        initial_distribution,
        "initial_distribution",
    )

    transition_matrix = _validate_stochastic_matrix(
        transition_matrix,
        "transition_matrix",
    )

    observation_matrix = _validate_stochastic_matrix(
        observation_matrix,
        "observation_matrix",
    )

    n_states = initial_distribution.size

    if transition_matrix.shape != (n_states, n_states):
        raise ValueError(
            "transition_matrix must have shape "
            f"({n_states}, {n_states}), but has "
            f"shape {transition_matrix.shape}."
        )

    if observation_matrix.shape[0] != n_states:
        raise ValueError(
            "observation_matrix must have one row for each hidden state."
        )

    if rng is None:
        rng = np.random.default_rng()

    n_observations = observation_matrix.shape[1]

    states = np.empty(length, dtype=np.int64)
    observations = np.empty(length, dtype=np.int64)

    # X_1
    states[0] = rng.choice(
        n_states,
        p=initial_distribution,
    )

    # Y_1 | X_1
    observations[0] = rng.choice(
        n_observations,
        p=observation_matrix[states[0]],
    )

    # X_t | X_{t-1}, followed by Y_t | X_t
    for t in range(1, length):
        states[t] = rng.choice(
            n_states,
            p=transition_matrix[states[t - 1]],
        )

        observations[t] = rng.choice(
            n_observations,
            p=observation_matrix[states[t]],
        )

    return states, observations


def binary_hmm_parameters(
    p: float,
    b: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Construct the binary HMM used in the paper's numerical experiments.

    Hidden-state space:
        X = {0, 1}

    Observation space:
        Y = {0, 1}

    Initial distribution:
        P(X_1 = 0) = P(X_1 = 1) = 1/2

    Transition matrix:
                [[p,     1-p],
            P =  [1-p,   p  ]]

    Observation matrix:
                [[b,     1-b],
            B =  [1-b,   b  ]]

    Parameters
    ----------
    p
        Probability of staying in the same hidden state.

    b
        Probability that the observation equals the hidden state.

    Returns
    -------
    initial_distribution, transition_matrix, observation_matrix
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1].")

    if not 0.0 <= b <= 1.0:
        raise ValueError("b must lie in [0, 1].")

    initial_distribution = np.array(
        [0.5, 0.5],
        dtype=float,
    )

    transition_matrix = np.array(
        [
            [p, 1.0 - p],
            [1.0 - p, p],
        ],
        dtype=float,
    )

    observation_matrix = np.array(
        [
            [b, 1.0 - b],
            [1.0 - b, b],
        ],
        dtype=float,
    )

    return (
        initial_distribution,
        transition_matrix,
        observation_matrix,
    )


def simulate_binary_hmm(
    p: float,
    b: float,
    length: int,
    seed: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Simulate the binary HMM used in the paper.

    Parameters
    ----------
    p
        Probability of remaining in the same hidden state.

    b
        Probability that Y_t = X_t.

    length
        Total sequence length.

    seed
        Random seed for reproducibility.

    Returns
    -------
    states, observations
    """
    (
        initial_distribution,
        transition_matrix,
        observation_matrix,
    ) = binary_hmm_parameters(p=p, b=b)

    rng = np.random.default_rng(seed)

    return simulate_hmm(
        initial_distribution=initial_distribution,
        transition_matrix=transition_matrix,
        observation_matrix=observation_matrix,
        length=length,
        rng=rng,
    )


def simulate_binary_experiment(
    p: float,
    b: float,
    T: int,
    T1: int,
    seed: int | None = None,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """
    Simulate one complete experiment consisting of:

        calibration period:  t = 1, ..., T
        prediction period:   t = T+1, ..., T+T1

    The entire path is generated first from one HMM and then split.

    Parameters
    ----------
    p
        Hidden-state persistence parameter.

    b
        Observation-accuracy parameter.

    T
        Calibration-sequence length.

    T1
        Prediction horizon.

    seed
        Random seed.

    Returns
    -------
    X_cal
        Hidden states during calibration.

    Y_cal
        Observations during calibration.

    X_future
        True hidden states during the prediction horizon.

    Y_future
        Future observations supplied to the prediction method.
    """
    if not isinstance(T, (int, np.integer)) or T <= 0:
        raise ValueError("T must be a positive integer.")

    if not isinstance(T1, (int, np.integer)) or T1 <= 0:
        raise ValueError("T1 must be a positive integer.")

    states, observations = simulate_binary_hmm(
        p=p,
        b=b,
        length=T + T1,
        seed=seed,
    )

    X_cal = states[:T]
    Y_cal = observations[:T]

    X_future = states[T:]
    Y_future = observations[T:]

    return X_cal, Y_cal, X_future, Y_future


def estimate_hmm_parameters(
    states: ArrayLike,
    observations: ArrayLike,
    n_states: int | None = None,
    n_observations: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate transition and observation matrices using empirical
    frequencies.

    This implements Eq. (12) of Nettasinghe et al. (2023).

    Transition estimate:

        P_hat[i, j]
            =
        number of transitions i -> j
        --------------------------------
        number of transitions leaving i

    Observation estimate:

        B_hat[i, y]
            =
        number of times X_t = i and Y_t = y
        ------------------------------------
        number of times X_t = i

    Parameters
    ----------
    states
        Hidden-state sequence.

    observations
        Observation sequence of the same length.

    n_states
        Number of hidden states. If None, inferred from `states`.

    n_observations
        Number of possible observations. If None, inferred from
        `observations`.

    Returns
    -------
    P_hat
        Estimated transition matrix.

    B_hat
        Estimated observation matrix.

    Notes
    -----
    The published estimator is undefined when a state has no outgoing
    transitions or never appears in the sequence. We deliberately do
    NOT add pseudocounts or smoothing here, because doing so would
    change the published method.

    Instead, this function raises a ValueError in such a case.
    """
    states = np.asarray(states, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.int64)

    if states.ndim != 1:
        raise ValueError("states must be one-dimensional.")

    if observations.ndim != 1:
        raise ValueError("observations must be one-dimensional.")

    if states.size != observations.size:
        raise ValueError(
            "states and observations must have the same length."
        )

    if states.size < 2:
        raise ValueError(
            "At least two time points are needed to estimate "
            "the transition matrix."
        )

    if np.any(states < 0):
        raise ValueError("State labels must be nonnegative integers.")

    if np.any(observations < 0):
        raise ValueError(
            "Observation labels must be nonnegative integers."
        )

    if n_states is None:
        n_states = int(states.max()) + 1

    if n_observations is None:
        n_observations = int(observations.max()) + 1

    if n_states <= 0:
        raise ValueError("n_states must be positive.")

    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")

    if states.max() >= n_states:
        raise ValueError(
            "states contains a label outside the specified state space."
        )

    if observations.max() >= n_observations:
        raise ValueError(
            "observations contains a label outside the specified "
            "observation space."
        )

    # -------------------------------------------------------------
    # Estimate transition matrix P_hat
    # -------------------------------------------------------------

    transition_counts = np.zeros(
        (n_states, n_states),
        dtype=float,
    )

    current_states = states[:-1]
    next_states = states[1:]

    np.add.at(
        transition_counts,
        (current_states, next_states),
        1.0,
    )

    outgoing_counts = np.bincount(
        current_states,
        minlength=n_states,
    ).astype(float)

    missing_transition_states = np.flatnonzero(
        outgoing_counts == 0
    )

    if missing_transition_states.size > 0:
        raise UndefinedHMMEstimateError(
            "Cannot estimate transition probabilities for state(s) "
            f"{missing_transition_states.tolist()} because they have "
            "no observed outgoing transitions."
        )

    P_hat = (
        transition_counts
        / outgoing_counts[:, None]
    )

    # -------------------------------------------------------------
    # Estimate observation matrix B_hat
    # -------------------------------------------------------------

    observation_counts = np.zeros(
        (n_states, n_observations),
        dtype=float,
    )

    np.add.at(
        observation_counts,
        (states, observations),
        1.0,
    )

    state_counts = np.bincount(
        states,
        minlength=n_states,
    ).astype(float)

    missing_observation_states = np.flatnonzero(
        state_counts == 0
    )

    if missing_observation_states.size > 0:
        raise UndefinedHMMEstimateError(
            "Cannot estimate observation probabilities for state(s) "
            f"{missing_observation_states.tolist()} because they never "
            "appear in the sequence."
        )

    B_hat = (
        observation_counts
        / state_counts[:, None]
    )

    return P_hat, B_hat


if __name__ == "__main__":
    p = 0.9
    b = 0.75
    T = 100
    T1 = 3
    seed = 42

    X_cal, Y_cal, X_future, Y_future = simulate_binary_experiment(
        p=p,
        b=b,
        T=T,
        T1=T1,
        seed=seed,
    )

    print("Calibration states:")
    print(X_cal[:20])

    print("\nCalibration observations:")
    print(Y_cal[:20])

    print("\nTrue future states:")
    print(X_future)

    print("\nFuture observations:")
    print(Y_future)

    P_hat, B_hat = estimate_hmm_parameters(
        states=X_cal,
        observations=Y_cal,
        n_states=2,
        n_observations=2,
    )

    print("\nEstimated transition matrix P_hat:")
    print(P_hat)

    print("\nEstimated observation matrix B_hat:")
    print(B_hat)