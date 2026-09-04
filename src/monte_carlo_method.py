from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


try:
    from .conformal import (
        filtering_conformity_score,
        identity_conformity_score,
        ordered_arrangement_count,
        terminal_window_from_blocks,
    )

    from .exact_method import (
        candidate_state_sequences,
        find_valid_block_family,
    )

    from .simulation import (
        estimate_hmm_parameters,
        UndefinedHMMEstimateError,
    )

except ImportError:
    from conformal import (
        filtering_conformity_score,
        identity_conformity_score,
        ordered_arrangement_count,
        terminal_window_from_blocks,
    )

    from exact_method import (
        candidate_state_sequences,
        find_valid_block_family,
    )

    from simulation import (
        estimate_hmm_parameters,
        UndefinedHMMEstimateError,
    )


@dataclass(frozen=True)
class MonteCarloCandidateResult:
    """
    Monte Carlo conformal result for one candidate sequence.
    """

    candidate: tuple[int, ...]

    q_hat: float | None

    included: bool

    identity_score: float | None

    n_blocks: int

    exact_permutation_count: int

    M: int

    successes: int

    block_pair: tuple[int, int] | None

    ij_index: int | None

    used_fallback: bool

    fallback_reason: str | None


@dataclass(frozen=True)
class MonteCarloPredictionResult:
    """
    Complete Monte Carlo conformal prediction result.
    """

    prediction_set: tuple[tuple[int, ...], ...]

    candidate_results: tuple[MonteCarloCandidateResult, ...]

    alpha: float

    horizon: int

    n_states: int

    M: int


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


def sample_uniform_block_arrangement(
    blocks,
    n_selected: int,
    rng: np.random.Generator,
):
    """
    Draw one arrangement uniformly from the ordered selections of
    `n_selected` distinct blocks.

    Suppose there are d blocks and r = n_selected.

    The exact permutation space contains

            d!
        -----------
        (d - r)!

    ordered arrangements.

    A uniformly random permutation of {0, ..., d-1}, followed by
    retaining its first r entries, generates every ordered selection
    with the same probability.

    Crucially, this does NOT construct the entire permutation space.

    Parameters
    ----------
    blocks
        Available exchangeable blocks.

    n_selected
        Number of distinct blocks in an arrangement.

        For prediction horizon T1,

            n_selected = T1 + 1.

    rng
        NumPy random-number generator.

    Returns
    -------
    arrangement
        Tuple containing the sampled blocks in their sampled order.
    """

    n_blocks = len(blocks)

    if not isinstance(
        n_selected,
        (int, np.integer),
    ):
        raise TypeError(
            "n_selected must be an integer."
        )

    if n_selected <= 0:
        raise ValueError(
            "n_selected must be positive."
        )

    if n_selected > n_blocks:
        raise ValueError(
            "Cannot select more distinct blocks than are available."
        )

    indices = rng.permutation(
        n_blocks
    )[:n_selected]

    return tuple(
        blocks[int(index)]
        for index in indices
    )


def monte_carlo_q_for_candidate(
    calibration_states: ArrayLike,
    calibration_observations: ArrayLike,
    future_observations: ArrayLike,
    candidate: ArrayLike,
    n_states: int,
    n_observations: int,
    M: int,
    rng: np.random.Generator,
) -> MonteCarloCandidateResult:
    """
    Estimate q(x) for one candidate by Monte Carlo sampling.

    The exact method would evaluate every element of Pi.

    This function instead draws M independent block arrangements
    uniformly from the same permutation space and computes

        q_hat_M(x)
            =
        (1 / M)
        sum_{m=1}^M
        1{S(Pi_m) >= S(I)}.

    Parameters
    ----------
    calibration_states
        X_1, ..., X_T.

    calibration_observations
        Y_1, ..., Y_T.

    future_observations
        Y_{T+1}, ..., Y_{T+T1}.

    candidate
        Candidate future hidden-state sequence.

    n_states
        Number of hidden states.

    n_observations
        Number of observation categories.

    M
        Number of independent permutation samples.

    rng
        NumPy random-number generator.

    Returns
    -------
    MonteCarloCandidateResult
        Candidate-level result containing q_hat_M and diagnostics.
    """

    if not isinstance(
        M,
        (int, np.integer),
    ):
        raise TypeError(
            "M must be an integer."
        )

    if M <= 0:
        raise ValueError(
            "M must be positive."
        )

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
            "calibration_observations contains an observation "
            "outside the specified observation space."
        )

    if np.any(Y_future >= n_observations):
        raise ValueError(
            "future_observations contains an observation outside "
            "the specified observation space."
        )

    horizon = x.size

    # -------------------------------------------------------------
    # Step 1:
    # Same augmentation as the exact method.
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
    # Same HMM parameter estimation as the exact method.
    # -------------------------------------------------------------

    try:
        P_hat, B_hat = estimate_hmm_parameters(
            states=X_augmented,
            observations=Y_augmented,
            n_states=n_states,
            n_observations=n_observations,
        )

    except UndefinedHMMEstimateError:

        return MonteCarloCandidateResult(
            candidate=tuple(
                int(v)
                for v in x
            ),
            q_hat=None,
            included=False,
            identity_score=None,
            n_blocks=0,
            exact_permutation_count=0,
            M=int(M),
            successes=0,
            block_pair=None,
            ij_index=None,
            used_fallback=True,
            fallback_reason="undefined_hmm_estimate",
        )

    # -------------------------------------------------------------
    # Step 3:
    # Same block family as the exact method.
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

        return MonteCarloCandidateResult(
            candidate=tuple(
                int(v)
                for v in x
            ),
            q_hat=None,
            included=False,
            identity_score=None,
            n_blocks=0,
            exact_permutation_count=0,
            M=int(M),
            successes=0,
            block_pair=None,
            ij_index=None,
            used_fallback=True,
            fallback_reason="insufficient_blocks",
        )

    # -------------------------------------------------------------
    # Size of the exact reduced permutation space.
    #
    # We compute only |Pi|. We never materialize Pi here.
    # -------------------------------------------------------------

    exact_permutation_count = ordered_arrangement_count(
        n_blocks=len(blocks),
        n_selected=horizon + 1,
    )

    # -------------------------------------------------------------
    # Same identity score S(I) as the exact method.
    # -------------------------------------------------------------

    S_identity = identity_conformity_score(
        states=X_augmented,
        observations=Y_augmented,
        horizon=horizon,
        transition_matrix=P_hat,
        observation_matrix=B_hat,
    )

    # -------------------------------------------------------------
    # Monte Carlo modification
    #
    # Exact method:
    #
    #     evaluate every pi in Pi
    #
    # Monte Carlo method:
    #
    #     draw Pi_1, ..., Pi_M iid from Uniform(Pi).
    # -------------------------------------------------------------

    successes = 0

    for _ in range(M):

        arrangement = sample_uniform_block_arrangement(
            blocks=blocks,
            n_selected=horizon + 1,
            rng=rng,
        )

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
            successes += 1

    # -------------------------------------------------------------
    # Monte Carlo estimator
    #
    #          successes
    # q_hat = -----------
    #              M
    # -------------------------------------------------------------

    q_hat = (
        successes
        / M
    )

    return MonteCarloCandidateResult(
        candidate=tuple(
            int(v)
            for v in x
        ),
        q_hat=float(q_hat),
        included=False,
        identity_score=float(S_identity),
        n_blocks=len(blocks),
        exact_permutation_count=exact_permutation_count,
        M=int(M),
        successes=int(successes),
        block_pair=pair,
        ij_index=ij_index,
        used_fallback=False,
        fallback_reason=None,
    )


def monte_carlo_prediction_set(
    calibration_states: ArrayLike,
    calibration_observations: ArrayLike,
    future_observations: ArrayLike,
    alpha: float,
    n_states: int,
    n_observations: int,
    M: int,
    seed: int | None = None,
) -> MonteCarloPredictionResult:
    """
    Construct the Monte Carlo approximation to the conformal
    prediction set.

    In this initial implementation, a candidate is included when

        q_hat_M(x) > alpha.

    No finite-M correction is applied here yet. That will be studied
    separately so that the raw Monte Carlo approximation can first be
    compared directly with the exact benchmark.

    Parameters
    ----------
    calibration_states
        Calibration hidden-state sequence.

    calibration_observations
        Calibration observation sequence.

    future_observations
        Future observations.

    alpha
        Miscoverage level.

    n_states
        Number of hidden states.

    n_observations
        Number of observation categories.

    M
        Number of sampled block arrangements per candidate.

    seed
        Random seed.

    Returns
    -------
    MonteCarloPredictionResult
        Approximate prediction set and candidate-level diagnostics.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must lie strictly between 0 and 1."
        )

    if not isinstance(
        M,
        (int, np.integer),
    ):
        raise TypeError(
            "M must be an integer."
        )

    if M <= 0:
        raise ValueError(
            "M must be positive."
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
        MonteCarloCandidateResult
    ] = []

    for candidate in candidate_state_sequences(
        n_states=n_states,
        horizon=horizon,
    ):

        raw_result = monte_carlo_q_for_candidate(
            calibration_states=calibration_states,
            calibration_observations=calibration_observations,
            future_observations=Y_future,
            candidate=np.asarray(
                candidate,
                dtype=np.int64,
            ),
            n_states=n_states,
            n_observations=n_observations,
            M=M,
            rng=rng,
        )

        # ---------------------------------------------------------
        # Standard Monte Carlo case
        # ---------------------------------------------------------

        if raw_result.q_hat is not None:

            included = (
                raw_result.q_hat > alpha
            )

        # ---------------------------------------------------------
        # Same practical randomized fallback as the exact benchmark
        # whenever q_hat cannot be constructed.
        # ---------------------------------------------------------

        else:

            included = bool(
                rng.random()
                <= (1.0 - alpha)
            )

        final_result = MonteCarloCandidateResult(
            candidate=raw_result.candidate,
            q_hat=raw_result.q_hat,
            included=included,
            identity_score=raw_result.identity_score,
            n_blocks=raw_result.n_blocks,
            exact_permutation_count=(
                raw_result.exact_permutation_count
            ),
            M=raw_result.M,
            successes=raw_result.successes,
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

    return MonteCarloPredictionResult(
        prediction_set=tuple(
            prediction_set
        ),
        candidate_results=tuple(
            candidate_results
        ),
        alpha=float(alpha),
        horizon=horizon,
        n_states=int(n_states),
        M=int(M),
    )


def monte_carlo_standard_error(
    q_hat: float,
    M: int,
) -> float:
    """
    Plug-in Monte Carlo standard error for q_hat_M.

    Since the Monte Carlo indicators are Bernoulli,

        Var(q_hat_M)
            =
        q(1-q) / M.

    Replacing q by q_hat gives

        SE_hat
            =
        sqrt(q_hat (1-q_hat) / M).
    """

    if not 0.0 <= q_hat <= 1.0:
        raise ValueError(
            "q_hat must lie in [0, 1]."
        )

    if not isinstance(
        M,
        (int, np.integer),
    ):
        raise TypeError(
            "M must be an integer."
        )

    if M <= 0:
        raise ValueError(
            "M must be positive."
        )

    return float(
        np.sqrt(
            q_hat
            * (1.0 - q_hat)
            / M
        )
    )


def sequence_is_covered(
    true_sequence: ArrayLike,
    prediction_set,
) -> bool:
    """
    Check whether the true future state sequence belongs to the
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
    Compute the scaled prediction-set size

            |C|
        ----------
         |X|^T1.
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
        from .exact_method import exact_prediction_set

    except ImportError:
        from simulation import simulate_binary_experiment
        from exact_method import exact_prediction_set

    p = 0.9
    b = 0.75

    T = 50
    T1 = 1

    alpha = 0.2

    data_seed = 42
    mc_seed = 123

    M = 1000

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
        seed=data_seed,
    )

    exact_result = exact_prediction_set(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        alpha=alpha,
        n_states=2,
        n_observations=2,
        seed=data_seed,
    )

    mc_result = monte_carlo_prediction_set(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        alpha=alpha,
        n_states=2,
        n_observations=2,
        M=M,
        seed=mc_seed,
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
        exact_result.prediction_set
    )

    print(
        "\nMonte Carlo prediction set:"
    )

    print(
        mc_result.prediction_set
    )

    print(
        "\nExact versus Monte Carlo candidate q-values:"
    )

    for exact_candidate, mc_candidate in zip(
        exact_result.candidate_results,
        mc_result.candidate_results,
    ):

        if mc_candidate.q_hat is not None:

            se = monte_carlo_standard_error(
                q_hat=mc_candidate.q_hat,
                M=M,
            )

        else:

            se = None

        if (
            exact_candidate.q_value is not None
            and mc_candidate.q_hat is not None
        ):

            absolute_error = abs(
                mc_candidate.q_hat
                - exact_candidate.q_value
            )

        else:

            absolute_error = None

        print(
            f"candidate={exact_candidate.candidate}, "
            f"q_exact={exact_candidate.q_value}, "
            f"q_hat={mc_candidate.q_hat}, "
            f"error={absolute_error}, "
            f"MC_SE={se}, "
            f"|Pi|={mc_candidate.exact_permutation_count}, "
            f"M={M}, "
            f"fallback={mc_candidate.used_fallback}, "
            f"fallback_reason={mc_candidate.fallback_reason}"
        )