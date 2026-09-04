from __future__ import annotations

from dataclasses import dataclass
from math import perm
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, order=True)
class Block:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Block start must be nonnegative.")

        if self.end <= self.start:
            raise ValueError(
                "Block end must be strictly greater than block start."
            )

    @property
    def length(self) -> int:
        """Number of elements contained in the block."""
        return self.end - self.start


def _as_integer_sequence(
    values: ArrayLike,
    name: str,
) -> NDArray[np.int64]:
    """
    Convert a one-dimensional sequence to a nonnegative integer array.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if values.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    # Permit things such as [0.0, 1.0] but reject non-integer labels.
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.equal(values, np.floor(values))):
            raise ValueError(
                f"{name} must contain integer-valued labels."
            )

    values = values.astype(np.int64)

    if np.any(values < 0):
        raise ValueError(
            f"{name} must contain nonnegative integer labels."
        )

    return values


def _validate_hmm_matrices(
    transition_matrix: ArrayLike,
    observation_matrix: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Validate an HMM transition matrix P and observation matrix B.
    """
    P = np.asarray(transition_matrix, dtype=float)
    B = np.asarray(observation_matrix, dtype=float)

    if P.ndim != 2:
        raise ValueError(
            "transition_matrix must be two-dimensional."
        )

    if P.shape[0] != P.shape[1]:
        raise ValueError(
            "transition_matrix must be square."
        )

    if B.ndim != 2:
        raise ValueError(
            "observation_matrix must be two-dimensional."
        )

    if B.shape[0] != P.shape[0]:
        raise ValueError(
            "observation_matrix must have one row for each hidden state."
        )

    if np.any(P < 0):
        raise ValueError(
            "transition_matrix cannot contain negative values."
        )

    if np.any(B < 0):
        raise ValueError(
            "observation_matrix cannot contain negative values."
        )

    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError(
            "Every row of transition_matrix must sum to 1."
        )

    if not np.allclose(B.sum(axis=1), 1.0):
        raise ValueError(
            "Every row of observation_matrix must sum to 1."
        )

    return P, B


def pair_occurrence_indices(
    states: ArrayLike,
    observations: ArrayLike,
    pair: tuple[int, int],
) -> NDArray[np.int64]:
    """
    Return all positions at which the augmented state (X_t, Y_t)
    equals a specified pair (i, j).

    Parameters
    ----------
    states
        Hidden-state sequence.

    observations
        Observation sequence.

    pair
        Pair (i, j).

    Returns
    -------
    indices
        Increasing array containing all t such that

            X_t = i
            Y_t = j.

    Notes
    -----
    Python uses zero-based indexing. This has no mathematical effect;
    it merely means that time t = 1 in the paper corresponds to
    array index 0 here.
    """
    X = _as_integer_sequence(states, "states")
    Y = _as_integer_sequence(observations, "observations")

    if X.size != Y.size:
        raise ValueError(
            "states and observations must have the same length."
        )

    i, j = pair

    if i < 0 or j < 0:
        raise ValueError(
            "The state-observation pair must be nonnegative."
        )

    return np.flatnonzero(
        (X == i) & (Y == j)
    ).astype(np.int64)


def find_complete_pair_blocks(
    states: ArrayLike,
    observations: ArrayLike,
    pair: tuple[int, int],
) -> list[Block]:
    """
    Construct all complete (i, j)-blocks for a specified pair.

    Suppose the augmented state (i, j) occurs at locations

        r_1 < r_2 < ... < r_d.

    Then the complete blocks are

        [r_1, r_2),
        [r_2, r_3),
        ...
        [r_{d-1}, r_d).

    Each block begins with (i, j) and contains no further (i, j).

    Parameters
    ----------
    states
        Hidden-state sequence.

    observations
        Observation sequence.

    pair
        Augmented state (i, j) defining the blocks.

    Returns
    -------
    blocks
        List of complete blocks.

    Notes
    -----
    If the pair occurs fewer than two times, there are no complete
    blocks and an empty list is returned.
    """
    indices = pair_occurrence_indices(
        states=states,
        observations=observations,
        pair=pair,
    )

    blocks = [
        Block(
            start=int(left),
            end=int(right),
        )
        for left, right in zip(
            indices[:-1],
            indices[1:],
        )
    ]

    return blocks


def terminal_pair(
    states: ArrayLike,
    observations: ArrayLike,
) -> tuple[int, int]:
    """
    Return the final augmented state (X_n, Y_n).

    In Algorithm 1, Step 3 constructs blocks corresponding to the
    state-observation pair appearing at the end of the augmented
    sequence.
    """
    X = _as_integer_sequence(states, "states")
    Y = _as_integer_sequence(observations, "observations")

    if X.size != Y.size:
        raise ValueError(
            "states and observations must have the same length."
        )

    return int(X[-1]), int(Y[-1])


def ordered_arrangement_count(
    n_blocks: int,
    n_selected: int,
) -> int:
    """
    Number of ordered selections of `n_selected` distinct blocks
    from `n_blocks`.

    This equals

        n_blocks!
        --------------------------
        (n_blocks - n_selected)!

    which is the permutation count used in the paper's numerical
    experiments.

    In our application,

        n_selected = T1 + 1.
    """
    if not isinstance(n_blocks, (int, np.integer)):
        raise TypeError(
            "n_blocks must be an integer."
        )

    if not isinstance(n_selected, (int, np.integer)):
        raise TypeError(
            "n_selected must be an integer."
        )

    if n_blocks < 0:
        raise ValueError(
            "n_blocks cannot be negative."
        )

    if n_selected < 0:
        raise ValueError(
            "n_selected cannot be negative."
        )

    if n_selected > n_blocks:
        return 0

    return perm(
        int(n_blocks),
        int(n_selected),
    )


def terminal_window_from_blocks(
    states: ArrayLike,
    observations: ArrayLike,
    blocks: Sequence[Block],
    window_length: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """
    Construct the final window produced by placing selected
    (i, j)-blocks in a specified order.

    The paper's filtering score depends only on the final T1 + 1
    augmented states. Therefore, when an ordered collection of blocks
    is specified, it is sufficient to concatenate those blocks and
    retain only the final T1 + 1 elements.

    Parameters
    ----------
    states
        Full augmented hidden-state sequence.

    observations
        Full augmented observation sequence.

    blocks
        Blocks listed in the desired order.

    window_length
        Number of final elements to retain.

        For prediction horizon T1:

            window_length = T1 + 1.

    Returns
    -------
    X_window
        Hidden-state component of the final window.

    Y_window
        Observation component of the final window.
    """
    X = _as_integer_sequence(states, "states")
    Y = _as_integer_sequence(observations, "observations")

    if X.size != Y.size:
        raise ValueError(
            "states and observations must have the same length."
        )

    if not isinstance(window_length, (int, np.integer)):
        raise TypeError(
            "window_length must be an integer."
        )

    if window_length <= 0:
        raise ValueError(
            "window_length must be positive."
        )

    if len(blocks) == 0:
        raise ValueError(
            "At least one block is required."
        )

    X_parts: list[NDArray[np.int64]] = []
    Y_parts: list[NDArray[np.int64]] = []

    for block in blocks:
        if not isinstance(block, Block):
            raise TypeError(
                "Every element of blocks must be a Block."
            )

        if block.end > X.size:
            raise ValueError(
                "A block extends beyond the supplied sequence."
            )

        X_parts.append(
            X[block.start:block.end]
        )

        Y_parts.append(
            Y[block.start:block.end]
        )

    X_concatenated = np.concatenate(X_parts)
    Y_concatenated = np.concatenate(Y_parts)

    if X_concatenated.size < window_length:
        raise ValueError(
            "The selected blocks do not contain enough elements "
            f"to construct a window of length {window_length}."
        )

    X_window = X_concatenated[-window_length:].copy()
    Y_window = Y_concatenated[-window_length:].copy()

    return X_window, Y_window


def hmm_filter(
    initial_state: int,
    future_observations: ArrayLike,
    transition_matrix: ArrayLike,
    observation_matrix: ArrayLike,
) -> NDArray[np.float64]:
    """
    Run the HMM filtering recursion used in Algorithm 1.

    Starting from a known hidden state X_T, the recursion is

        p_{T+k}
            =
        B_{Y_{T+k}} P^T p_{T+k-1}
        --------------------------------
        1^T B_{Y_{T+k}} P^T p_{T+k-1},

    where

        p_{T+k}[i]
            =
        P(
            X_{T+k} = i
            |
            X_T,
            Y_{T+1}, ..., Y_{T+k}
        ).

    Parameters
    ----------
    initial_state
        Known state at the beginning of the window.

    future_observations
        Sequence

            Y_{T+1}, ..., Y_{T+T1}.

    transition_matrix
        Estimated HMM transition matrix P_hat.

    observation_matrix
        Estimated observation matrix B_hat.

    Returns
    -------
    filtered_probabilities
        Array of shape

            (T1, n_states).

        Row k contains the filtering distribution after observing
        the first k + 1 future observations.
    """
    P, B = _validate_hmm_matrices(
        transition_matrix,
        observation_matrix,
    )

    Y_future = _as_integer_sequence(
        future_observations,
        "future_observations",
    )

    n_states = P.shape[0]
    n_observations = B.shape[1]

    if not isinstance(initial_state, (int, np.integer)):
        raise TypeError(
            "initial_state must be an integer."
        )

    if initial_state < 0 or initial_state >= n_states:
        raise ValueError(
            "initial_state is outside the HMM state space."
        )

    if np.any(Y_future >= n_observations):
        raise ValueError(
            "future_observations contains a label outside the "
            "observation space."
        )

    # At time T, X_T is assumed known.
    posterior = np.zeros(
        n_states,
        dtype=float,
    )

    posterior[int(initial_state)] = 1.0

    filtered_probabilities = np.empty(
        (Y_future.size, n_states),
        dtype=float,
    )

    for k, observation in enumerate(Y_future):

        # Prediction:
        #
        # P(X_{t+1} | observations up to t)
        #
        predicted = P.T @ posterior

        # Update using Y_{t+1}.
        #
        # Multiplication by diag(B[:, y]) is equivalent to
        # elementwise multiplication by B[:, y].
        #
        unnormalized = (
            B[:, int(observation)] * predicted
        )

        normalizing_constant = unnormalized.sum()

        if (
            normalizing_constant <= 0.0
            or not np.isfinite(normalizing_constant)
        ):
            raise FloatingPointError(
                "The HMM filtering update has zero or non-finite "
                "normalizing probability. The observed sequence has "
                "zero probability under the supplied HMM parameters."
            )

        posterior = (
            unnormalized
            / normalizing_constant
        )

        filtered_probabilities[k] = posterior

    return filtered_probabilities


def filtering_conformity_score(
    state_window: ArrayLike,
    observation_window: ArrayLike,
    transition_matrix: ArrayLike,
    observation_matrix: ArrayLike,
) -> float:
    """
    Compute the filtering-based conformity score S for one window.

    The window has the form

        X_T, X_{T+1}, ..., X_{T+T1}

    together with

        Y_T, Y_{T+1}, ..., Y_{T+T1}.

    The paper defines

        S
        =
        1
        -
        (1 / T1)
        sum_{k=1}^{T1}
        P(
            X_{T+k}
            |
            X_T,
            Y_{T+1}, ..., Y_{T+k}
        ).

    Larger values therefore represent worse agreement between the
    candidate state sequence and the HMM/filtering probabilities.

    Parameters
    ----------
    state_window
        Sequence of length T1 + 1.

    observation_window
        Observation sequence of the same length.

    transition_matrix
        P_hat.

    observation_matrix
        B_hat.

    Returns
    -------
    score
        Filtering conformity score S.
    """
    X = _as_integer_sequence(
        state_window,
        "state_window",
    )

    Y = _as_integer_sequence(
        observation_window,
        "observation_window",
    )

    if X.size != Y.size:
        raise ValueError(
            "state_window and observation_window must have "
            "the same length."
        )

    if X.size < 2:
        raise ValueError(
            "A conformity-score window must contain at least "
            "one conditioning state and one predicted state."
        )

    P, B = _validate_hmm_matrices(
        transition_matrix,
        observation_matrix,
    )

    n_states = P.shape[0]
    n_observations = B.shape[1]

    if np.any(X >= n_states):
        raise ValueError(
            "state_window contains a state outside the HMM state space."
        )

    if np.any(Y >= n_observations):
        raise ValueError(
            "observation_window contains an observation outside "
            "the HMM observation space."
        )

    T1 = X.size - 1

    filtered_probabilities = hmm_filter(
        initial_state=int(X[0]),
        future_observations=Y[1:],
        transition_matrix=P,
        observation_matrix=B,
    )

    # At step k, obtain the filtering probability assigned to the
    # candidate state X_{T+k}.
    candidate_probabilities = filtered_probabilities[
        np.arange(T1),
        X[1:],
    ]

    score = (
        1.0
        - candidate_probabilities.mean()
    )

    return float(score)


def identity_conformity_score(
    states: ArrayLike,
    observations: ArrayLike,
    horizon: int,
    transition_matrix: ArrayLike,
    observation_matrix: ArrayLike,
) -> float:
    """
    Compute S(I), the conformity score of the unpermuted augmented
    sequence.

    Since the filtering score depends only on the last T1 + 1
    elements, S(I) is obtained directly from the terminal window

        t = T, ..., T + T1.

    Parameters
    ----------
    states
        Full augmented hidden-state sequence.

    observations
        Full augmented observation sequence.

    horizon
        Prediction horizon T1.

    transition_matrix
        P_hat.

    observation_matrix
        B_hat.

    Returns
    -------
    score
        Identity conformity score S(I).
    """
    X = _as_integer_sequence(
        states,
        "states",
    )

    Y = _as_integer_sequence(
        observations,
        "observations",
    )

    if X.size != Y.size:
        raise ValueError(
            "states and observations must have the same length."
        )

    if not isinstance(horizon, (int, np.integer)):
        raise TypeError(
            "horizon must be an integer."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    window_length = horizon + 1

    if X.size < window_length:
        raise ValueError(
            "The augmented sequence is shorter than T1 + 1."
        )

    X_window = X[-window_length:]
    Y_window = Y[-window_length:]

    return filtering_conformity_score(
        state_window=X_window,
        observation_window=Y_window,
        transition_matrix=transition_matrix,
        observation_matrix=observation_matrix,
    )


if __name__ == "__main__":

    P = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.75, 0.25],
            [0.25, 0.75],
        ],
        dtype=float,
    )

    X = np.array(
        [0, 1, 0, 0, 1, 0, 1, 0],
        dtype=int,
    )

    Y = np.array(
        [0, 1, 0, 1, 1, 0, 0, 0],
        dtype=int,
    )

    print("Terminal augmented pair:")
    print(terminal_pair(X, Y))

    pair = terminal_pair(X, Y)

    blocks = find_complete_pair_blocks(
        states=X,
        observations=Y,
        pair=pair,
    )

    print("\nComplete blocks for terminal pair:")
    print(blocks)

    print("\nNumber of ordered selections of 2 blocks:")
    print(
        ordered_arrangement_count(
            n_blocks=len(blocks),
            n_selected=2,
        )
    )

    filtered = hmm_filter(
        initial_state=0,
        future_observations=np.array([1, 0]),
        transition_matrix=P,
        observation_matrix=B,
    )

    print("\nFiltering probabilities:")
    print(filtered)

    score = filtering_conformity_score(
        state_window=np.array([0, 1, 0]),
        observation_window=np.array([0, 1, 0]),
        transition_matrix=P,
        observation_matrix=B,
    )

    print("\nExample conformity score:")
    print(score)