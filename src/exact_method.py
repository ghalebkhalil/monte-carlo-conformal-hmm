from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from numpy.typing import ArrayLike, NDArray


try:
    from .conformal import (
        Block,
        filtering_conformity_score,
        find_complete_pair_blocks,
        identity_conformity_score,
        ordered_arrangement_count,
        terminal_window_from_blocks,
    )

    from .simulation import (
        estimate_hmm_parameters,
        UndefinedHMMEstimateError,
    )

except ImportError:
    from conformal import (
        Block,
        filtering_conformity_score,
        find_complete_pair_blocks,
        identity_conformity_score,
        ordered_arrangement_count,
        terminal_window_from_blocks,
    )

    from simulation import (
        estimate_hmm_parameters,
        UndefinedHMMEstimateError,
    )


@dataclass(frozen=True)
class ExactCandidateResult:
    """
    Exact conformal result for one candidate future state sequence.
    """

    candidate: tuple[int, ...]

    q_value: float | None

    included: bool

    identity_score: float | None

    n_blocks: int

    n_permutations: int

    block_pair: tuple[int, int] | None

    ij_index: int | None

    used_fallback: bool

    fallback_reason: str | None


@dataclass(frozen=True)
class ExactPredictionResult:
    """
    Complete exact conformal-prediction result.
    """

    prediction_set: tuple[tuple[int, ...], ...]

    candidate_results: tuple[ExactCandidateResult, ...]

    alpha: float

    horizon: int

    n_states: int


def _as_integer_sequence(
    values: ArrayLike,
    name: str,
) -> NDArray[np.int64]:
    """
    Convert input to a one-dimensional nonnegative integer array.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        if not np.all(
            np.equal(
                values,
                np.floor(values),
            )
        ):
            raise ValueError(
                f"{name} must contain integer-valued labels."
            )

    values = values.astype(
        np.int64
    )

    if np.any(values < 0):
        raise ValueError(
            f"{name} must contain nonnegative labels."
        )

    return values


def candidate_state_sequences(
    n_states: int,
    horizon: int,
):
    """
    Generate all candidate hidden-state sequences in X^{T1}.

    For example, with two states and T1 = 2:

        (0, 0)
        (0, 1)
        (1, 0)
        (1, 1)

    The number of candidates is

        n_states ** horizon.
    """

    if not isinstance(
        n_states,
        (int, np.integer),
    ):
        raise TypeError(
            "n_states must be an integer."
        )

    if not isinstance(
        horizon,
        (int, np.integer),
    ):
        raise TypeError(
            "horizon must be an integer."
        )

    if n_states <= 0:
        raise ValueError(
            "n_states must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    return product(
        range(int(n_states)),
        repeat=int(horizon),
    )


def find_valid_block_family(
    states: ArrayLike,
    observations: ArrayLike,
    horizon: int,
) -> tuple[
    tuple[int, int] | None,
    list[Block],
    int | None,
]:
    """
    Find a block family with enough blocks to form arrangements of
    length T1 + 1.

    The authors' released implementation searches

        ij_index = 1, 2, ..., T1

    where ij_index = 1 corresponds to the final augmented pair,

        (X_n, Y_n),

    ij_index = 2 corresponds to

        (X_{n-1}, Y_{n-1}),

    and so forth.

    The first pair producing at least T1 + 1 complete blocks is used.

    Parameters
    ----------
    states
        Augmented hidden-state sequence.

    observations
        Augmented observation sequence.

    horizon
        Prediction horizon T1.

    Returns
    -------
    pair
        Selected (i, j) pair, or None if no valid family exists.

    blocks
        Complete blocks corresponding to that pair.

    ij_index
        1-based distance from the end used to select the pair.

        For example:

            ij_index = 1  -> final pair
            ij_index = 2  -> one pair earlier
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

    if not isinstance(
        horizon,
        (int, np.integer),
    ):
        raise TypeError(
            "horizon must be an integer."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    required_blocks = horizon + 1

    maximum_index = min(
        horizon,
        X.size,
    )

    for ij_index in range(
        1,
        maximum_index + 1,
    ):

        pair = (
            int(X[-ij_index]),
            int(Y[-ij_index]),
        )

        blocks = find_complete_pair_blocks(
            states=X,
            observations=Y,
            pair=pair,
        )

        if len(blocks) >= required_blocks:
            return (
                pair,
                blocks,
                ij_index,
            )

    return (
        None,
        [],
        None,
    )


def exact_q_for_candidate(
    calibration_states: ArrayLike,
    calibration_observations: ArrayLike,
    future_observations: ArrayLike,
    candidate: ArrayLike,
    n_states: int,
    n_observations: int,
) -> ExactCandidateResult:
    """
    Compute the exact conformal q-value for one candidate future
    hidden-state sequence.

    This function implements Steps 1--5 of Algorithm 1 for one
    candidate sequence.

    Parameters
    ----------
    calibration_states
        X_1, ..., X_T.

    calibration_observations
        Y_1, ..., Y_T.

    future_observations
        Y_{T+1}, ..., Y_{T+T1}.

    candidate
        Candidate hidden-state sequence

            x_{T+1}, ..., x_{T+T1}.

    n_states
        Cardinality |X| of the hidden-state space.

    n_observations
        Cardinality |Y| of the observation space.

    Returns
    -------
    ExactCandidateResult
        Contains the exact q-value and associated diagnostics.

    Notes
    -----
    q_value is returned as None whenever the exact procedure cannot
    be evaluated because:

        - the empirical HMM estimate is undefined, or
        - no suitable block family exists.

    The prediction-set function then applies the practical randomized
    fallback and records the precise reason.
    """

    X_cal = _as_integer_sequence(
        calibration_states,
        "calibration_states",
    )

    Y_cal = _as_integer_sequence(
        calibration_observations,
        "calibration_observations",
    )

    Y_future = _as_integer_sequence(
        future_observations,
        "future_observations",
    )

    x = _as_integer_sequence(
        candidate,
        "candidate",
    )

    if X_cal.size != Y_cal.size:
        raise ValueError(
            "Calibration states and observations must have "
            "the same length."
        )

    if x.size != Y_future.size:
        raise ValueError(
            "candidate and future_observations must have "
            "the same length."
        )

    if n_states <= 0:
        raise ValueError(
            "n_states must be positive."
        )

    if n_observations <= 0:
        raise ValueError(
            "n_observations must be positive."
        )

    if np.any(X_cal >= n_states):
        raise ValueError(
            "calibration_states contains a state outside "
            "the specified state space."
        )

    if np.any(x >= n_states):
        raise ValueError(
            "candidate contains a state outside "
            "the specified state space."
        )

    if np.any(Y_cal >= n_observations):
        raise ValueError(
            "calibration_observations contains a label outside "
            "the specified observation space."
        )

    if np.any(Y_future >= n_observations):
        raise ValueError(
            "future_observations contains a label outside "
            "the specified observation space."
        )

    horizon = x.size

    # -------------------------------------------------------------
    # Step 1:
    # Augment the calibration state sequence by assuming that the
    # candidate is the true future hidden-state sequence.
    # -------------------------------------------------------------

    X_augmented = np.concatenate(
        [
            X_cal,
            x,
        ]
    )

    Y_augmented = np.concatenate(
        [
            Y_cal,
            Y_future,
        ]
    )

    # -------------------------------------------------------------
    # Step 2:
    # Estimate P_hat and B_hat from the complete augmented sequence.
    # -------------------------------------------------------------

    try:
        P_hat, B_hat = estimate_hmm_parameters(
            states=X_augmented,
            observations=Y_augmented,
            n_states=n_states,
            n_observations=n_observations,
        )

    except UndefinedHMMEstimateError:

        return ExactCandidateResult(
            candidate=tuple(
                int(v)
                for v in x
            ),
            q_value=None,
            included=False,
            identity_score=None,
            n_blocks=0,
            n_permutations=0,
            block_pair=None,
            ij_index=None,
            used_fallback=True,
            fallback_reason="undefined_hmm_estimate",
        )

    # -------------------------------------------------------------
    # Step 3:
    # Find a suitable family of exchangeable (i, j)-blocks.
    # -------------------------------------------------------------

    (
        pair,
        blocks,
        ij_index,
    ) = find_valid_block_family(
        states=X_augmented,
        observations=Y_augmented,
        horizon=horizon,
    )

    if pair is None:

        return ExactCandidateResult(
            candidate=tuple(
                int(v)
                for v in x
            ),
            q_value=None,
            included=False,
            identity_score=None,
            n_blocks=0,
            n_permutations=0,
            block_pair=None,
            ij_index=None,
            used_fallback=True,
            fallback_reason="insufficient_blocks",
        )

    # -------------------------------------------------------------
    # Number of exact block arrangements:
    #
    #           d!
    #     ----------------
    #     (d - T1 - 1)!
    #
    # -------------------------------------------------------------

    n_permutations = ordered_arrangement_count(
        n_blocks=len(blocks),
        n_selected=horizon + 1,
    )

    # -------------------------------------------------------------
    # Identity score S(I)
    # -------------------------------------------------------------

    S_identity = identity_conformity_score(
        states=X_augmented,
        observations=Y_augmented,
        horizon=horizon,
        transition_matrix=P_hat,
        observation_matrix=B_hat,
    )

    # -------------------------------------------------------------
    # Step 4 + Step 5:
    #
    # Enumerate every member of the reduced permutation space.
    #
    # We generate permutations lazily instead of materializing the
    # entire permutation list in memory.
    # -------------------------------------------------------------

    count_at_least_identity = 0
    count_total = 0

    arrangements = permutations(
        blocks,
        horizon + 1,
    )

    for arrangement in arrangements:

        X_window, Y_window = terminal_window_from_blocks(
            states=X_augmented,
            observations=Y_augmented,
            blocks=arrangement,
            window_length=horizon + 1,
        )

        S_pi = filtering_conformity_score(
            state_window=X_window,
            observation_window=Y_window,
            transition_matrix=P_hat,
            observation_matrix=B_hat,
        )

        if S_pi >= S_identity:
            count_at_least_identity += 1

        count_total += 1

    # -------------------------------------------------------------
    # Defensive checks
    # -------------------------------------------------------------

    if count_total != n_permutations:
        raise RuntimeError(
            "The number of enumerated permutations does not match "
            "the theoretical ordered-arrangement count."
        )

    if count_total == 0:
        raise RuntimeError(
            "A valid block family was selected but produced "
            "zero permutations."
        )

    # -------------------------------------------------------------
    # Exact q(x)
    # -------------------------------------------------------------

    q_value = (
        count_at_least_identity
        / count_total
    )

    return ExactCandidateResult(
        candidate=tuple(
            int(v)
            for v in x
        ),
        q_value=float(q_value),
        included=False,
        identity_score=float(S_identity),
        n_blocks=len(blocks),
        n_permutations=n_permutations,
        block_pair=pair,
        ij_index=ij_index,
        used_fallback=False,
        fallback_reason=None,
    )


def exact_prediction_set(
    calibration_states: ArrayLike,
    calibration_observations: ArrayLike,
    future_observations: ArrayLike,
    alpha: float,
    n_states: int,
    n_observations: int,
    seed: int | None = None,
) -> ExactPredictionResult:
    """
    Construct the exact conformal prediction set.

    A candidate x is included when

        q(x) > alpha.

    When q(x) cannot be evaluated, we retain the practical randomized
    fallback used for the benchmark implementation:

        include the candidate with probability 1 - alpha.

    The reason for the fallback is recorded separately as either

        "undefined_hmm_estimate"

    or

        "insufficient_blocks".

    Parameters
    ----------
    calibration_states
        Calibration hidden states.

    calibration_observations
        Calibration observations.

    future_observations
        Future observations over prediction horizon T1.

    alpha
        Miscoverage level.

    n_states
        Number of hidden states.

    n_observations
        Number of observation categories.

    seed
        Seed used only for fallback randomization.

    Returns
    -------
    ExactPredictionResult
        Prediction set and detailed result for every candidate.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must lie strictly between 0 and 1."
        )

    Y_future = _as_integer_sequence(
        future_observations,
        "future_observations",
    )

    horizon = Y_future.size

    rng = np.random.default_rng(
        seed
    )

    prediction_set: list[
        tuple[int, ...]
    ] = []

    candidate_results: list[
        ExactCandidateResult
    ] = []

    for candidate in candidate_state_sequences(
        n_states=n_states,
        horizon=horizon,
    ):

        raw_result = exact_q_for_candidate(
            calibration_states=calibration_states,
            calibration_observations=calibration_observations,
            future_observations=Y_future,
            candidate=np.asarray(
                candidate,
                dtype=np.int64,
            ),
            n_states=n_states,
            n_observations=n_observations,
        )

        # ---------------------------------------------------------
        # Normal exact case
        # ---------------------------------------------------------

        if raw_result.q_value is not None:

            included = (
                raw_result.q_value > alpha
            )

        # ---------------------------------------------------------
        # Practical randomized fallback
        # ---------------------------------------------------------

        else:

            included = bool(
                rng.random()
                <= (1.0 - alpha)
            )

        final_result = ExactCandidateResult(
            candidate=raw_result.candidate,
            q_value=raw_result.q_value,
            included=included,
            identity_score=raw_result.identity_score,
            n_blocks=raw_result.n_blocks,
            n_permutations=raw_result.n_permutations,
            block_pair=raw_result.block_pair,
            ij_index=raw_result.ij_index,
            used_fallback=raw_result.used_fallback,
            fallback_reason=raw_result.fallback_reason,
        )

        candidate_results.append(
            final_result
        )

        if included:
            prediction_set.append(
                final_result.candidate
            )

    return ExactPredictionResult(
        prediction_set=tuple(
            prediction_set
        ),
        candidate_results=tuple(
            candidate_results
        ),
        alpha=float(alpha),
        horizon=horizon,
        n_states=int(n_states),
    )


def sequence_is_covered(
    true_sequence: ArrayLike,
    prediction_set,
) -> bool:
    """
    Check whether the true hidden-state sequence belongs to a
    prediction set.
    """

    true_sequence = _as_integer_sequence(
        true_sequence,
        "true_sequence",
    )

    true_tuple = tuple(
        int(v)
        for v in true_sequence
    )

    return (
        true_tuple
        in prediction_set
    )


def scaled_prediction_set_size(
    prediction_set,
    n_states: int,
    horizon: int,
) -> float:
    """
    Compute

            |C_{1-alpha}|
        ---------------------
              |X| ^ T1

    which is the scaled prediction-set size reported in the paper.
    """

    if n_states <= 0:
        raise ValueError(
            "n_states must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    total_candidates = (
        n_states ** horizon
    )

    return float(
        len(prediction_set)
        / total_candidates
    )


if __name__ == "__main__":

    try:
        from .simulation import simulate_binary_experiment

    except ImportError:
        from simulation import simulate_binary_experiment

    # Small test so exact enumeration remains fast.

    p = 0.9
    b = 0.75

    T = 50
    T1 = 1

    alpha = 0.2
    seed = 42

    (
        X_cal,
        Y_cal,
        X_future,
        Y_future,
    ) = simulate_binary_experiment(
        p=p,
        b=b,
        T=T,
        T1=T1,
        seed=seed,
    )

    result = exact_prediction_set(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        alpha=alpha,
        n_states=2,
        n_observations=2,
        seed=seed,
    )

    print(
        "True future sequence:"
    )

    print(
        tuple(
            int(v)
            for v in X_future
        )
    )

    print(
        "\nFuture observations:"
    )

    print(
        tuple(
            int(v)
            for v in Y_future
        )
    )

    print(
        "\nExact prediction set:"
    )

    print(
        result.prediction_set
    )

    print(
        "\nCandidate diagnostics:"
    )

    for candidate_result in result.candidate_results:

        print(
            f"candidate={candidate_result.candidate}, "
            f"q={candidate_result.q_value}, "
            f"included={candidate_result.included}, "
            f"blocks={candidate_result.n_blocks}, "
            f"|Pi|={candidate_result.n_permutations}, "
            f"pair={candidate_result.block_pair}, "
            f"ij_index={candidate_result.ij_index}, "
            f"fallback={candidate_result.used_fallback}, "
            f"fallback_reason={candidate_result.fallback_reason}"
        )

    covered = sequence_is_covered(
        true_sequence=X_future,
        prediction_set=result.prediction_set,
    )

    scaled_size = scaled_prediction_set_size(
        prediction_set=result.prediction_set,
        n_states=2,
        horizon=T1,
    )

    print(
        "\nTrue sequence covered:"
    )

    print(
        covered
    )

    print(
        "\nScaled prediction-set size:"
    )

    print(
        scaled_size
    )